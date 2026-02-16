import os
from unittest.mock import MagicMock, patch
import utils_bigquery

def test_get_full_table_id_signatures():
    print("Testing get_full_table_id signatures...")
    # Test default
    default_id = utils_bigquery.get_full_table_id()
    print(f"Default ID: {default_id}")
    assert default_id == "cal-rag-agent.calrag.documents"
    
    # Test overrides
    override_id = utils_bigquery.get_full_table_id(project="p", dataset="d", table="t")
    print(f"Override ID: {override_id}")
    assert override_id == "p.d.t"
    print("get_full_table_id tests passed!")

@patch('utils_bigquery.get_bq_client')
def test_insert_rows_json_print_fix(mock_get_client):
    print("\nTesting insert_rows_json print fix...")
    # Mock client and insert_rows_json method
    mock_client_instance = MagicMock()
    mock_get_client.return_value = mock_client_instance
    mock_client_instance.insert_rows_json.return_value = [] # No errors
    
    # Call insert_rows_json should not raise AttributeError
    try:
        utils_bigquery.insert_rows_json([{"test": "data"}], project="test_p", dataset="test_d", table="test_t")
        print("insert_rows_json executed successfully without AttributeError.")
    except AttributeError as e:
        print(f"FAILED: AttributeError caught: {e}")
        raise
    except Exception as e:
        print(f"FAILED: Verify script had unexpected error: {e}")
        raise

if __name__ == "__main__":
    test_get_full_table_id_signatures()
    test_insert_rows_json_print_fix()
    print("\nVerification successful!")
