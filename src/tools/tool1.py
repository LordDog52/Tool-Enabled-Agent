from psycopg_pool import ConnectionPool
from psycopg import sql
import os
from dotenv import load_dotenv

from datetime import datetime, timezone

def get_schema_metadata(schema: str = "intern_task") -> list[dict]:
    """
    Retrieve database schema metadata and map PostgreSQL datatypes to Python types.

    This function queries PostgreSQL's information_schema to obtain all tables,
    columns, and their corresponding datatypes within the specified schema.
    PostgreSQL datatypes are converted into equivalent Python datatype names
    (e.g., integer → int, text → str, ARRAY → list).

    Args:
        schema (str, optional):
            Name of the database schema to inspect.
            Defaults to "intern_task".

    Returns:
        list[dict]:
            A list of dictionaries, each representing a table and its columns.

            Format:
            [
                {
                    "table": "accounts",
                    "columns": [
                        ["user_id", "str"],
                        ["name", "str"],
                        ["last_login", "datetime"]
                    ]
                },
                ...
            ]

    Raises:
        psycopg.Error:
            If database connection or query execution fails.

    Example:
        >>> metadata = get_schema_metadata()
        >>> metadata[0]
        {
            "table": "accounts",
            "columns": [["user_id", "str"], ["name", "str"], ...]
        }

    Notes:
        - ARRAY types are mapped to Python 'list'
        - TIMESTAMPTZ and TIMESTAMP are mapped to Python 'datetime'
        - This metadata is used for input validation and dynamic query building.
    """
    PG_TO_PYTHON_TYPE = {
        "smallint": "int",
        "integer": "int",
        "bigint": "int",

        "decimal": "float",
        "numeric": "float",
        "real": "float",
        "double precision": "float",

        "character varying": "str",
        "character": "str",
        "text": "str",

        "boolean": "bool",

        "timestamp without time zone": "datetime",
        "timestamp with time zone": "datetime",
        "date": "date",
        "time without time zone": "time",

        "ARRAY": "list",

        "json": "dict",
        "jsonb": "dict",

        "uuid": "str"
    }

    query = """
        SELECT
            table_name,
            column_name,
            data_type
        FROM information_schema.columns
        WHERE table_schema = %s
        ORDER BY table_name, ordinal_position
    """

    schema_dict = {}

    with pool.connection() as conn:
        with conn.cursor() as cur:

            cur.execute(query, (schema,))
            rows = cur.fetchall()

            for table, column, data_type in rows:

                python_type = PG_TO_PYTHON_TYPE.get(data_type, "Any")

                if table not in schema_dict:
                    schema_dict[table] = {
                        "table": table,
                        "columns": []
                    }

                schema_dict[table]["columns"].append(
                    [column, python_type]
                )

    return list(schema_dict.values())


def get_value(data, key):
    """
    Get value for certain key in dictionary
    """
    for k, v in data:
        if k == key:
            return v
    return None

def get_columns(schema, table_name):
    """
    Retrieve all column names for a specific table.

    Args:
        schema (list[dict]):
            Schema metadata from get_schema_metadata().

        table_name (str):
            Name of the table.

    Returns:
        list[str]:
            List of column names.

    Example:
        >>> get_columns(metadata, "accounts")
        ['user_id', 'name', 'role', 'status', 'service_plan', 'last_login']
    """
    for table in schema:
        if table["table"] == table_name:
            return [column[0] for column in table["columns"]]

    return []

def get_datatype(schema, table_name, column_name):
    """
    Retrieve the Python datatype for a specific table column from schema metadata.

    Args:
        schema (list[dict]):
            Schema metadata from get_schema_metadata().

        table_name (str):
            Name of the table.

        column_name (str):
            Name of the column.

    Returns:
        str:
            Python datatype name (e.g., 'int', 'str', 'datetime', 'list').

        Returns empty list if not found.

    Example:
        >>> get_datatype(metadata, "accounts", "last_login")
        'datetime'
    """
    for table in schema:
        if table["table"] == table_name:
            for i in [column for column in table["columns"]]:
                if i[0] == column_name:
                    return i[1]
    return []

def can_convert_to_int(value) -> bool:
    """
    Check if a value can be converted to an integer.
    """
    try:
        int(value)
        return True
    except (ValueError, TypeError):
        return False
    
def validate_input(metadata, table: str, filters: dict) -> bool:
    """
    Validate structured query parameters against database schema metadata.

    This function ensures that the provided table name and filter conditions
    are valid according to the database schema metadata. It verifies:

        - The table parameter is a string.
        - The filters parameter is a dictionary.
        - The specified table exists in the schema.
        - All provided column names exist in the table.
        - Each filter value matches the expected Python datatype derived
          from PostgreSQL column types.
        - String values for datetime columns are convertible to ISO format.
        - String values for integer columns are convertible to int.
        - ARRAY (list) columns accept string input for ANY comparison.

    If a datetime value is provided as an ISO-format string, it will be
    automatically converted to a `datetime` object in-place.

    Args:
        metadata (list[dict]):
            Schema metadata returned from `get_schema_metadata()`.

        table (str):
            Name of the database table to query.

        filters (dict):
            Dictionary of column-value pairs used for filtering.

            Example:
                {
                    "user_id": "1001",
                    "last_login": "2026-02-17T10:15:00+00:00"
                }

    Returns:
        bool:
            True if validation passes successfully.

    Raises:
        TypeError:
            - If `table` is not a string.
            - If `filters` is not a dictionary.
            - If column names are not strings.
            - If a value does not match expected datatype.

        ValueError:
            - If the specified table does not exist.
            - If a specified column does not exist in the table.

    Notes:
        - Integer columns accept numeric strings convertible to int.
        - Datetime columns accept ISO 8601 formatted strings.
        - ARRAY columns allow string input for PostgreSQL ANY matching.
        - This function mutates `filters` when converting datetime strings.
    """

    # Validate table and filters types
    if type(table) != str or type(filters) != dict:
        if type(table) != str:
            raise TypeError(f"table should be of type str, but got {type(table).__name__}")
        else:
            raise TypeError(f"filters should be of type dict, but got {type(filters).__name__}")

    # Validate column types
    for column in filters:
        if type(column) != str:
            raise TypeError(f"Column names should be of type str, but got {type(column).__name__}")

    # Validate is table exists or not
    if table not in [table['table'] for table in metadata]: # Extract table names from metadata
        raise ValueError(f"Table {table} does not exist in the database schema, available table {[table['table'] for table in metadata]}")

    # Validate is column exists or not
    for i in filters.keys():
        if i not in get_columns(metadata, table):
            raise ValueError(f"Column {i} does not exist in table {table}, available column {get_columns(metadata, table)}")
    
    # validate values types
    for column, value in filters.items():
        if type(value) == str and get_datatype(metadata, table, column) == "datetime":
            try:
                filters[column] = datetime.fromisoformat(value)
                continue
            except ValueError:
                raise TypeError(f"Value for column {column} should be of type datetime or a string in ISO format, but got '{value}' which cannot be parsed as datetime")

        if type(value) == str and get_datatype(metadata, table, column) == "int":
            if not can_convert_to_int(value):
                raise TypeError(f"Value for column {column} should be of type int or a string that can be converted to int, but got '{value}' which cannot be converted to int")
            else:
                continue
        if type(value) == str and get_datatype(metadata, table, column) == "list":
            continue
        if type(value).__name__ != get_datatype(metadata, table, column):
            raise TypeError(f"Value for column {column} should be of type {get_datatype(metadata, table, column)}, but got {type(value).__name__}")
        
    return True

def Structured_Data_Query_Tool(table: str, filters: dict) -> list[tuple]:
    """
    Execute a validated, structured SELECT query against the database.

    This function dynamically builds and executes a parameterized SQL query
    based on a target table and filter conditions. Input is validated against
    database schema metadata before query execution.

    Supported features:
    - Automatic schema validation
    - Dynamic WHERE clause construction
    - ARRAY column filtering using PostgreSQL ANY operator
    - Multi-column filtering using AND conditions

    Args:
        table (str):
            Name of the database table to query.

        filters (dict):
            Dictionary of column-value pairs used for filtering.
            If filters is an empty dictionary, the query returns all rows
            from the table (no WHERE clause).

            Example:
                {
                    "user_id": "1001"
                }

            Multiple filters:
                {
                    "user_id": "1001",
                    "status": "Active"
                }

    Returns:
        list[tuple]:
            Query results as a list of database rows.

            Example:
                [
                    ("1001", "Alice Tan", "Employee", "Active", "Basic Support", datetime(...))
                ]

    Raises:
        ValueError:
            If table or column does not exist.

        TypeError:
            If filter values do not match expected datatypes.

        Exception:
            If schema metadata retrieval fails.

        psycopg.Error:
            If SQL execution fails.
                Database Schema:
    
    Database schema:

        dataset_metadata:
            id: int
            version: str
            last_updated: datetime
            description: str

        policies:
            policy_id: str
            title: str
            category: str
            description: str
            role_scope: str

        policy_rules:
            policy_id: str
            rule_order: int
            rule_text: str

        sla_lookup:
            service_name: str
            tier: str
            response_time: str
            resolution_time: str
            availability: str
            support_channels: str
            escalation_available: bool

        accounts:
            user_id: str
            name: str
            role: str
            status: str
            service_plan: str
            last_login: datetime

        system_status:
            id: int
            current_load_percentage: int
            active_incidents: int
            system_health: str
            maintenance_mode: bool
            last_updated: datetime
    """

    conditions = []
    values = []
    try:
        metadata = get_schema_metadata()
    except Exception as e:
        raise Exception(f"Failed to retrieve schema metadata: {str(e)}")

    validate_input(metadata, table, filters)

    for column, value in filters.items():
        if get_datatype(metadata, table, column) == "list":
            conditions.append(
                sql.SQL("%s = ANY({})").format(sql.Identifier(column))
            )
        else:
            conditions.append(
                sql.SQL("{} = %s").format(sql.Identifier(column))
            )

        values.append(value)
    
    if conditions:
        where_clause = sql.SQL(" AND ").join(conditions)
    else:
        where_clause = None
    
    if where_clause:
        query = sql.SQL("""
            SELECT *
            FROM intern_task.{table}
            WHERE {where}
        """).format(
            table=sql.Identifier(table),
            where=where_clause
        )
    else:
        query = sql.SQL("""
            SELECT *
            FROM intern_task.{table}
        """).format(
            table=sql.Identifier(table)
        )

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, values)
            return cur.fetchall()



if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Construct the database URL from environment variables
    DB_URL = (
        f"postgresql://{os.getenv('DB_USER')}:"
        f"{os.getenv('DB_PASSWORD')}@"
        f"{os.getenv('DB_HOST')}:"
        f"{os.getenv('DB_PORT')}/"
        f"{os.getenv('DB_NAME')}"
    )

    # Initialize the connection pool
    pool = ConnectionPool(DB_URL)

    # Get schema metadata
    metadata = get_schema_metadata()

    # Example usage of the Structured_Data_Query_Tool
    table = 'accounts'
    filters = {}
    print(Structured_Data_Query_Tool(table, filters))