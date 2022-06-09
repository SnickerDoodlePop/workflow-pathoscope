import json
import shutil
from pathlib import Path
from shutil import copytree
from types import SimpleNamespace

import pytest
import virtool_workflow.execution.run_subprocess
from virtool_workflow.analysis.analysis import Analysis
from virtool_workflow.analysis.indexes import Index
from virtool_workflow.analysis.library_types import LibraryType
from virtool_workflow.analysis.reads import Reads
from virtool_workflow.data_model import NucleotideComposition, Subtraction
from virtool_workflow.data_model.samples import Sample

from pathoscope import parse_sam
from workflow import (
    eliminate_subtraction,
    map_default_isolates,
    map_isolates,
    reassignment,
)

TEST_DATA_PATH = Path(__file__).parent / "test_files"
FASTQ_PATH = TEST_DATA_PATH / "test.fq"
INDEX_PATH = TEST_DATA_PATH / "index"
SUBTRACTION_PATH = TEST_DATA_PATH / "subtraction"
SAM_PATH = TEST_DATA_PATH / "test_al.sam"


@pytest.fixture
def work_path(tmpdir):
    return Path(tmpdir)


@pytest.fixture
def index(work_path: Path):
    index_path = work_path / "indexes/index3"
    shutil.copytree(INDEX_PATH, index_path)

    return Index(
        id="index3",
        manifest={
            "foobar": 10,
            "reo": 5,
            "baz": 6,
        },
        _sequence_otu_map={
            "NC_016509": "foobar",
            "NC_001948": "foobar",
            "13TF149_Reovirus_TF1_Seg06": "reo",
            "13TF149_Reovirus_TF1_Seg03": "reo",
            "13TF149_Reovirus_TF1_Seg07": "reo",
            "13TF149_Reovirus_TF1_Seg02": "reo",
            "13TF149_Reovirus_TF1_Seg08": "reo",
            "13TF149_Reovirus_TF1_Seg11": "reo",
            "13TF149_Reovirus_TF1_Seg04": "reo",
            "NC_004667": "foobar",
            "NC_003347": "foobar",
            "NC_003615": "foobar",
            "NC_003689": "foobar",
            "NC_011552": "foobar",
            "KX109927": "baz",
            "NC_008039": "foobar",
            "NC_015782": "foobar",
            "NC_016416": "foobar",
            "NC_003623": "foobar",
            "NC_008038": "foobar",
            "NC_001836": "foobar",
            "JQ080272": "baz",
            "NC_017938": "foobar",
            "NC_008037": "foobar",
            "NC_007448": "foobar",
        },
        reference=None,
        path=index_path,
        ready=True,
        _run_in_executor=run_in_executor,
        _run_subprocess=run_subprocess,
    )


@pytest.fixture
def sample(work_path: Path):
    shutil.copyfile(FASTQ_PATH, work_path / "reads_1.fq.gz")
    sample_ = Sample(
        id="foobar",
        name="foobar",
        paired=False,
        library_type=LibraryType.other,
        quality={"count": 1337, "length": [78, 101]},
        locale=None,
        isolate=None,
        host=None,
    )

    sample_.read_paths = (work_path / "reads_1.fq.gz",)

    return sample_


@pytest.fixture
def read_file_names(sample):
    return ",".join(str(p) for p in sample.read_paths)


@pytest.fixture
def reads(work_path: Path):
    shutil.copyfile(FASTQ_PATH, work_path / "reads_1.fq.gz")
    return Reads(sample=sample, quality={}, path=work_path)


@pytest.fixture
def subtraction(work_path: Path):
    subtractions_path = work_path / "subtractions"
    subtractions_path.mkdir(parents=True)

    subtraction_path = work_path / "subtractions" / "subtraction"

    copytree(SUBTRACTION_PATH, subtraction_path)

    nucleotide_composition = NucleotideComposition(
        a=0.1,
        t=0.2,
        g=0.3,
        c=0.4,
    )

    return Subtraction(
        id="arabidopsis_thaliana",
        name="Arabidopsis thaliana",
        nickname="Thalecress",
        count=12,
        gc=nucleotide_composition,
        path=subtraction_path,
    )


@pytest.fixture
async def run_in_executor():
    async def _run_in_executor(func, *args):
        return func(*args)

    return _run_in_executor


@pytest.fixture
async def run_subprocess():
    return virtool_workflow.execution.run_subprocess.run_subprocess()


@pytest.fixture
def ref_lengths():
    with (TEST_DATA_PATH / "ref_lengths.json").open("r") as f:
        return json.load(f)


async def test_map_default_isolates(
    read_file_names, index: Index, run_subprocess, snapshot
):
    intermediate = SimpleNamespace(to_otus=set())

    await map_default_isolates(
        intermediate, read_file_names, index, 2, 0.01, run_subprocess
    )

    assert sorted(intermediate.to_otus) == snapshot


async def test_map_isolates(
    index,
    read_file_names,
    work_path,
    run_subprocess,
    snapshot,
):
    for path in INDEX_PATH.iterdir():
        if "reference" in path.name:
            shutil.copyfile(
                path, work_path / path.name.replace("reference", "isolates")
            )

    intermediate = SimpleNamespace(isolate_high_scores={})
    isolate_fastq_path = work_path / "mapped.fq"
    isolate_sam_path = work_path / "to_isolates.sam"

    await map_isolates(
        read_file_names,
        intermediate,
        isolate_fastq_path,
        work_path / "isolates",
        isolate_sam_path,
        run_subprocess,
        1,
        0.01,
    )

    with isolate_sam_path.open("r") as f:
        assert sorted(line.rstrip() for line in f) == snapshot

    assert intermediate.isolate_high_scores == snapshot


@pytest.mark.datafiles(SAM_PATH, FASTQ_PATH)
async def test_eliminate_subtraction(datafiles, subtraction, work_path, run_subprocess):
    isolate_fastq_path = work_path / "test.fq"
    isolate_sam_path = work_path / "test_al.sam"

    subtracted_path = work_path / "subtracted.sam"

    results = {}

    await eliminate_subtraction(
        isolate_fastq_path,
        isolate_sam_path,
        2,
        results,
        run_subprocess,
        subtraction,
        subtracted_path,
        work_path,
    )

    assert not (work_path / "to_subtraction.sam").is_file()
    assert (work_path / "subtracted.sam").is_file()
    assert results["subtracted_count"] == 4


async def test_pathoscope(
    mocker,
    data_regression,
    file_regression,
    index,
    ref_lengths,
    run_in_executor,
    snapshot,
    work_path: Path,
):
    subtracted_path = work_path / "subtracted.sam"
    shutil.copyfile(SAM_PATH, subtracted_path)

    analysis = mocker.Mock(spec=Analysis)
    intermediate = SimpleNamespace(lengths=ref_lengths)
    results = {}

    await reassignment(
        analysis,
        index,
        intermediate,
        0.01,
        results,
        run_in_executor,
        subtracted_path,
        work_path,
    )

    data_regression.check(
        sorted(
            [
                (line.read_id, line.ref_id, line.score)
                for line in parse_sam(intermediate.reassigned_path)
            ]
        )
    )

    with intermediate.report_path.open("r") as f:
        file_regression.check(f.read(), extension=".tsv")
