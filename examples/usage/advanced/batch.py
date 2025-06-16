import pixeltable as pxt
from pixelmemory import Memory
import datetime

custom_schema = {
    'case_id': pxt.Int,
    'customer_name': pxt.String,
    'product': pxt.String,
    'issue_type': pxt.String,
    'status': pxt.String,
    'priority': pxt.String,
    'created_at': pxt.Timestamp,
    'resolved': pxt.Bool,
    'order_details': pxt.Json,
}

memory = Memory(
    namespace='customer_service',
    table_name='customer_support_agent',
    schema=custom_schema,
    if_exists='replace_force',
)

entries = [
    {
        'content': "Customer is unable to log in to their account. They've tried resetting their password but "
        "haven't received the reset email. They are using the correct email address associated with the account.",
        'metadata': {
            'case_id': 1,
            'customer_name': 'Alice Johnson',
            'product': 'Pixel-Auth',
            'issue_type': 'Login Issue',
            'status': 'Open',
            'priority': 'High',
            'created_at': datetime.datetime.now(),
            'resolved': False,
            'order_details': {'order_id': None, 'items': []},
        },
    },
    {
        'content': "The user's recent order (Order #ORD-552) arrived with a damaged component. They have attached a "
        'photo of the damaged item and are requesting a replacement part.',
        'metadata': {
            'case_id': 2,
            'customer_name': 'Bob Williams',
            'product': 'Pixel-Gadget',
            'issue_type': 'Damaged Product',
            'status': 'In Progress',
            'priority': 'Medium',
            'created_at': datetime.datetime.now() - datetime.timedelta(days=1),
            'resolved': False,
            'order_details': {'order_id': 'ORD-552', 'items': ['Pixel-Gadget']},
        },
    },
    {
        'content': 'A customer is inquiring about the return policy for a product they purchased last week. They want to know if they can get a full refund and what the process is.',
        'metadata': {
            'case_id': 3,
            'customer_name': 'Charlie Brown',
            'product': 'Pixel-Store',
            'issue_type': 'Billing Inquiry',
            'status': 'Resolved',
            'priority': 'Low',
            'created_at': datetime.datetime.now() - datetime.timedelta(days=5),
            'resolved': True,
            'order_details': {'order_id': 'ORD-540', 'items': ['Pixel-Accessory']},
        },
    },
]

memory.add_entries(entries)