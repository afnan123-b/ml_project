"""
Script to run the training pipeline for the model.
"""

from src.data.module.random_forest import main




processed_columns = [
    'Item_Weight',
    'Item_Visibility',
    'Item_MRP',
    'Outlet_Establishment_Year',
    'Item_Identifier_encoded',
    'Item_Type_encoded',
    'Item_Fat_Content_encoded_LF',
    'Item_Fat_Content_encoded_REG',
    'Outlet_Location_Type_Tier 1',
    'Outlet_Location_Type_Tier 2',
    'Outlet_Location_Type_Tier 3',
    'Outlet_Size_High',
    'Outlet_Size_Medium',
    'Outlet_Size_Small',
    'Outlet_Type_Grocery Store',
    'Outlet_Type_Supermarket Type1',
    'Outlet_Type_Supermarket Type2',
    'Outlet_Type_Supermarket Type3',
    'Sales'
]

if __name__ == "__main__":
    file_path = "data/raw/processed/train.csv"
    target_column = "Sales"

    print("🚀 Running training pipeline...")
    main(file_path, target_column, processed_columns)
