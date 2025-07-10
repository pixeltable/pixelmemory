from pixelmemory import Memory

mem = Memory(namespace="my_app_memory", table_name="user_data", if_exists="ignore")

print(f"Successfully created memory table at: {mem.table.memory_id}")

print(f"Table Information: {mem.table.get_metadata()}")
