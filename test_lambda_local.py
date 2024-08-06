import json
from lambda_function import lambda_handler
from example_events import events


def test_lambda_function():
    for i, event in enumerate(events):
        print(f"\nTesting event {i + 1}:")
        context = {}
        response = lambda_handler(event, context)

        print("Input:")
        print(json.loads(event['body'])['document_text'][:100] + "...")

        print("\nOutput:")
        print(json.dumps(response, indent=2))

        print("\n" + "=" * 50)


if __name__ == "__main__":
    test_lambda_function()