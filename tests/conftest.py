def pytest_addoption(parser):
    parser.addoption(
        "--eval",
        action="store_true",
        default=False,
        help="Run live LLM evals against examples/",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "eval: live LLM replay of example scripts (enable with --eval)",
    )
