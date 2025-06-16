import pixeltable as pxt
from pixelmemory import Memory

custom_schema = {
    'account_id': pxt.String,
    'account_name': pxt.String,
    'account_type': pxt.String,
    'account_balance': pxt.Float,
}

memory = Memory(
    namespace='customer_service',
    table_name='customer_support_agent',
    schema=custom_schema,
    if_exists='ignore',
)

print(memory.table.get_metadata())