"""Integer arithmetic addition unit type 'I2x7O8'"""

import csv
import hashlib
import json
import logging
from collections import namedtuple
from datetime import datetime, timezone
from pathlib import Path

from pydantic import Field

from ..common import int_to_bitstring, bitstring_to_trits, trits_to_bitstring, bitstring64_to_int, random_trit_vector
from ..dataset import ITUInputDatasetMeta, dataset_meta_csv
from ..exceptions import AigarthITUError, AigarthITUInputDataError, PYDANTIC_ERRORS
from ..itu_cl import AigarthITUCl, ITUReflection
from ..itu_perf import ITUHitFactorsBin, ITUHitFactorsBinAggregate
from .common import CAPABILITY, aigarth_itu

LOG = logging.getLogger(Path(__file__).stem)


AAIInputDatasetRow = namedtuple("AAIInputDatasetRow", ("a", "b", "c"))


class ITUReflectionArithmeticAdditionInt(ITUReflection):
    """Arithmetic addition reflection."""

    bitstring: str | None = Field(default=None, frozen=True)
    unk_count: int | None = Field(default=None, frozen=True)
    integer: int | None = Field(default=None, frozen=True)

    def __init__(self, **kwargs):
        """Initialize instance of arithmetic addition reflection."""
        if trits := kwargs.get("trits"):
            bitstring = trits_to_bitstring(trits=trits)
            unk_count = bitstring.count("?")
            kwargs["bitstring"] = bitstring
            kwargs["unk_count"] = unk_count
            kwargs["integer"] = bitstring64_to_int(bit_str=bitstring) if unk_count == 0 else None

        super().__init__(**kwargs)


@aigarth_itu
class ITUClArithmeticAdditionIntI2x7O8(AigarthITUCl):
    """Integer arithmetic addition unit type 'I2x7O8'.

    Input:      A,B: 2x 7bit signed integers ([-64, +63])
    Operation:  "+" (arithmetic addition)
    Output:     C (=A+B): 1x 8bit signed integer ([-128, +127])
    """

    NEURON_INPUT_COUNT_INIT = 200
    NEURON_INPUT_SKEW_INIT = 0
    ITU_INPUT_BITWIDTH = 14
    ITU_OUTPUT_BITWIDTH = 8

    def __init__(self) -> None:
        """Initialize ITU."""
        super().__init__(
            itu_group=CAPABILITY.MATH.ADDITION,
            input_bitwidth=self.ITU_INPUT_BITWIDTH,
            output_bitwidth=self.ITU_OUTPUT_BITWIDTH,
        )
        # Tune up neuron inputs
        for n in self._circle:
            n.input_weights = random_trit_vector(size=self.NEURON_INPUT_COUNT_INIT)
            n.input_skew = self.NEURON_INPUT_SKEW_INIT

    def reflect(self, a: int, b: int) -> ITUReflectionArithmeticAdditionInt:
        """Figure out result of arithmetic addition of input arguments.

        :param a:   operand 'a'
        :param b:   operand 'b'
        :return:    input 'reflection'
        """
        lm_prefix = f"{self.lm_prefix}Reflect: "

        try:
            a_bitstring = int_to_bitstring(num=a, max_bits=7, const_len=True)
            b_bitstring = int_to_bitstring(num=b, max_bits=7, const_len=True)
            a_trits = bitstring_to_trits(bitstring=a_bitstring)
            b_trits = bitstring_to_trits(bitstring=b_bitstring)
            input_trits = a_trits + b_trits
        except (TypeError, ValueError) as e:
            raise AigarthITUInputDataError(f"{lm_prefix}Prepare ITU 'feed': {type(e).__name__}: {e}") from e

        forward, ffcycle_stats = self.feedforward(feed=input_trits)

        try:
            reflection = ITUReflectionArithmeticAdditionInt(
                trits=forward,
                ffcycle_stats=ffcycle_stats,
            )
        except (TypeError, ValueError) + PYDANTIC_ERRORS as e:
            raise AigarthITUError(f"{lm_prefix}Create reflection: {type(e).__name__}: {e}") from e

        return reflection

    def reflect_many(
        self, dataset: list[AAIInputDatasetRow] | None = None, dataset_fpath: Path | None = None
    ) -> list[tuple[ITUReflectionArithmeticAdditionInt, int]]:
        """Apply an intelligent capability to a set of input objects.

        :param dataset:         input dataset object. Takes precedence over dataset_fpath.
        :param dataset_fpath:   input dataset file path
        :return:                set of ITU reflection object + corresponding 'gage' value pairs
        """
        assert dataset is not None or dataset_fpath is not None, "Either 'dataset' or 'dataset_fpath' must be provided."

        LOG.info(
            f"{self.lm_prefix}Create reflections: itu_id={self.meta.id}, itu_ver={self.meta.version_string}, "
            f"dataset={dataset_fpath or f'{type(dataset)} at {id(dataset)}'} ..."
        )

        csv_dh = None  # bound before the try so the finally's close() can't mask a dataset error
        try:
            if dataset_fpath:
                ids_metadata = dataset_meta_csv(dataset_fpath, self.meta.tds_hash_algorithm)
                csv_dh = open(dataset_fpath, newline="")
                _ = [csv_dh.readline() for i in range(ids_metadata.metadata_size)]  # skip to data start
            else:
                ids_digest = hashlib.new(
                    self.meta.tds_hash_algorithm, json.dumps([i._asdict() for i in dataset]).encode()
                ).hexdigest()
                ids_metadata = ITUInputDatasetMeta(
                    data_type=self.meta.group,
                    data_source="runtime",
                    id=ids_digest,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    record_fields=" ".join(AAIInputDatasetRow._fields),
                    record_count=len(dataset),
                    digest=ids_digest,
                )
                csv_dh = [f"{ids_metadata.field_delimiter}".join([str(f) for f in row]) for row in dataset]

            ids_rows = csv.DictReader(
                f=csv_dh, fieldnames=ids_metadata.record_fields, delimiter=ids_metadata.field_delimiter
            )

            LOG.info(
                f"{self.lm_prefix}Create reflections: itu_id={self.meta.id}, itu_ver={self.meta.version_string}, "
                f"dataset_meta={ids_metadata}"
            )

            # Walk through input data records
            reflection_eds = []
            for ids_row in ids_rows:
                reflection = self.reflect(a=int(ids_row["a"]), b=int(ids_row["b"]))
                reflection_eds.append((reflection, int(ids_row["c"])))
                LOG.debug(
                    f"{self.lm_prefix}Create reflections: itu_id={self.meta.id}, itu_ver={self.meta.version_string}, "
                    f"{ids_row=}, {reflection=}, gage_value{int(ids_row['c'])}"
                )
        except (OSError, TypeError, ValueError) + PYDANTIC_ERRORS as e:
            raise AigarthITUError(f"{self.lm_prefix}Create reflections: {type(e).__name__}: {e}") from e
        finally:
            if dataset_fpath and csv_dh is not None:
                csv_dh.close()

        LOG.info(
            f"{self.lm_prefix}Create reflections: itu_id={self.meta.id}, itu_ver={self.meta.version_string}, "
            f"dataset={dataset_fpath or f'{type(dataset)} at {id(dataset)}'}: OK"
        )

        return reflection_eds

    def hit_factors(self, reflection: ITUReflectionArithmeticAdditionInt, gage: int) -> ITUHitFactorsBin:
        """Compute reflection 'hit factors'.

        :param reflection:  source reflection object
        :param gage:        gage (reference) value
        :return:            reflection evaluation data
        """
        try:
            gage_bitstring = int_to_bitstring(num=gage, max_bits=self.ITU_OUTPUT_BITWIDTH, const_len=True)
        except ValueError as e:
            raise AigarthITUError(f"{self.lm_prefix}Compute reflection 'hit factors': {type(e).__name__}: {e}") from e

        hitbit_count = 0
        for g, r in zip(gage_bitstring, reflection.bitstring):
            if g == r:
                hitbit_count += 1

        hit_factors = ITUHitFactorsBin(
            gage_bitstring=gage_bitstring,
            hitbit_count=hitbit_count,
            unk_count=reflection.unk_count,
            ffcycle_stats=reflection.ffcycle_stats,
        )

        return hit_factors

    def hit_factors_aggregate(
        self, eval_ds: list[tuple[ITUReflectionArithmeticAdditionInt, int]]
    ) -> ITUHitFactorsBinAggregate:
        """Compute aggregated reflection 'hit factors'.

        :param eval_ds: source dataset for evaluation
        :return:        aggregated reflection evaluation data
        """
        # aggregated reflection stats
        ar_stats = dict(allbit_count=0, hitbit_count=0, unk_count=0, alltick_count=0)
        # Walk through input data items
        for ein_item in eval_ds:
            hit_factors = self.hit_factors(reflection=ein_item[0], gage=ein_item[1])
            # Update ITU aggregated reflection stats
            ar_stats["allbit_count"] += self.ITU_OUTPUT_BITWIDTH
            ar_stats["hitbit_count"] += hit_factors.hitbit_count
            ar_stats["unk_count"] += hit_factors.unk_count
            ar_stats["alltick_count"] += hit_factors.ffcycle_stats.tick_count

        itu_rhf_aggregate = ITUHitFactorsBinAggregate(**ar_stats)

        return itu_rhf_aggregate
