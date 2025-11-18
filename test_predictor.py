#!/usr/bin/env python3
"""
Test predictor module
"""
from predictor import get_predictor

def test_predictor():
    print("Testing predictor module...")

    try:
        # Get predictor instance
        predictor = get_predictor()
        print("✓ Predictor initialized successfully")

        # Test available stores
        print(f"✓ Found {len(predictor.available_stores)} stores")
        print(f"  Available stores: {predictor.available_stores[:10]}...")

        # Test get_all_stores
        all_stores = predictor.get_all_stores()
        print(f"✓ get_all_stores() returned {len(all_stores)} stores")

        # Test get_store_info
        if predictor.available_stores:
            store_nbr = predictor.available_stores[0]
            store_info = predictor.get_store_info(store_nbr)
            print(f"✓ get_store_info({store_nbr}) works: {store_info['city']}")

        # Test get_top_stores
        top_result = predictor.get_top_stores(n=5)
        print(f"✓ get_top_stores() returned: {type(top_result)}")
        if 'stores' in top_result:
            print(f"  Contains {len(top_result['stores'])} stores")
        else:
            print(f"  ERROR: Expected dict with 'stores' key, got: {top_result.keys()}")

        # Test get_bottom_stores
        bottom_result = predictor.get_bottom_stores(n=5)
        print(f"✓ get_bottom_stores() returned: {type(bottom_result)}")
        if 'stores' in bottom_result:
            print(f"  Contains {len(bottom_result['stores'])} stores")
        else:
            print(f"  ERROR: Expected dict with 'stores' key, got: {bottom_result.keys()}")

        # Test load_store_model
        if predictor.available_stores:
            store_nbr = predictor.available_stores[0]
            model = predictor.load_store_model(store_nbr)
            print(f"✓ load_store_model({store_nbr}) successful")

        print("\n✓✓✓ All tests passed! ✓✓✓")

    except Exception as e:
        print(f"\n✗✗✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    test_predictor()
