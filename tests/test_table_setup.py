import pytest
import pixeltable as pxt
from pixelmemory import Memory

# Clean up any existing test directories or tables before and after tests
# to ensure test isolation.
@pytest.fixture(autouse=True)
def cleanup_pixeltable():
    test_namespace = "test_pytest_namespace"
    # These are the 'table_name' arguments passed to Memory() in the tests.
    table_user_names = ["test_pytest_table", "test_custom_schema_table"]
    
    # Construct the full names of tables created in the root by the Memory class.
    root_table_names_to_clean = [
        f"{test_namespace}_{user_name}" for user_name in table_user_names
    ]

    def _perform_cleanup():
        # 1. Drop the root-level tables created by Memory instances.
        for table_name in root_table_names_to_clean:
            pxt.drop_table(table_name, if_exists=True, force=True)
        
        # 2. Drop the directory created by Memory instances.
        # This directory is expected to be empty if Memory class only creates root tables.
        pxt.drop_dir(test_namespace, if_exists=True, force=True)

    _perform_cleanup() # Pre-test cleanup: run before each test
    yield              # This is where the test itself runs
    _perform_cleanup() # Post-test cleanup: run after each test


def test_memory_table_creation():
    test_namespace = "test_pytest_namespace"
    test_table_name_user = "test_pytest_table"
    
    Memory(
        namespace=test_namespace,
        table_name=test_table_name_user
    )

    assert test_namespace in pxt.list_dirs(), \
        f"Namespace '{test_namespace}' was not found in pixeltable directories."
    assert f"{test_namespace}_{test_table_name_user}" in pxt.list_tables(), \
        f"Table '{test_namespace}_{test_table_name_user}' was not found in pixeltable tables."

def test_memory_table_creation_with_custom_schema():
    test_namespace = "test_pytest_namespace"
    test_table_name_user = "test_custom_schema_table"
    custom_schema = {
        'custom_field1': pxt.String,
        'custom_field2': pxt.Int,
    }
    
    Memory(
        namespace=test_namespace,
        table_name=test_table_name_user,
        schema=custom_schema
    )

    assert test_namespace in pxt.list_dirs(), \
        f"Namespace '{test_namespace}' was not found in pixeltable directories."
    assert f"{test_namespace}_{test_table_name_user}" in pxt.list_tables(), \
        f"Table '{test_namespace}_{test_table_name_user}' was not found in pixeltable tables."
    
    Memory(
        namespace=test_namespace,
        table_name=test_table_name_user,
        schema=custom_schema
    )

    assert test_namespace in pxt.list_dirs(), \
        f"Namespace '{test_namespace}' was not found for custom schema test."

    expected_table_name_internal = f"{test_namespace}_{test_table_name_user}"
    assert expected_table_name_internal in pxt.list_tables(), \
        f"Table '{expected_table_name_internal}' was not found for custom schema test."

    # Optionally, verify the schema of the created table
    table_handle = pxt.get_table(expected_table_name_internal)
    table_schema = table_handle.schema()
    
    # Check for default fields
    assert 'memory_id' in table_schema
    assert 'insert_at' in table_schema
    assert 'content' in table_schema
    assert 'metadata' in table_schema
    
    # Check for custom fields
    assert 'custom_field1' in table_schema
    assert table_schema['custom_field1'].type == pxt.StringType()
    assert 'custom_field2' in table_schema
    assert table_schema['custom_field2'].type == pxt.IntType()