import pixeltable as pxt
from pixelmemory import Memory

schema = {
    'case_id': pxt.Int,
    'status': pxt.String,
}

memory = Memory(
    namespace='customer_service',
    table_name='customer_support_agent',
    schema=schema,
    if_exists='replace_force',
)

memory_id = memory.add_entry(
    content="Customer is unable to log in to their account. They've tried resetting their password but "
    "haven't received the reset email. They are using the correct email address associated with the account.",
    metadata={
        'case_id': 1,
        'status': 'Open',
    },
)

print(memory.table.select().collect())