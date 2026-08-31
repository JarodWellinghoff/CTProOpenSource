from ctpro.cli import build_parser


def test_cli_parser_builds_with_expected_program_name() -> None:
    parser = build_parser()
    assert parser.prog == "ctpro"
    help_text = parser.format_help()
    for command in ("nps", "cho", "mtf", "mtfc"):
        assert command in help_text
