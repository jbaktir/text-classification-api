from example_events import events
from lambda_function import lambda_handler
event = events[0]
context = ''
lambda_handler(event, context)
