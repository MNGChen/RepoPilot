"""Entry point for the small repository DevPilot will inspect."""

from src.greetings import build_greeting


def main() -> None:
    """Run the sample application."""
    print(build_greeting("DevPilot"))


if __name__ == "__main__":
    main()
