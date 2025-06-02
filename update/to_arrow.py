import json
import pyarrow as pa
import pyarrow.feather as feather # For writing Feather files (a common Arrow on-disk format)
import argparse
from tqdm import tqdm # For progress bar

def extract_data_from_json_line(line_str):
    """
    Parses a single JSON line and extracts data into a flat dictionary.
    """
    try:
        record = json.loads(line_str)
    except json.JSONDecodeError:
        print(f"Warning: Skipping malformed JSON line: {line_str[:100]}...")
        return None

    features_data = record.get("features")
    if not features_data:
        print(f"Warning: Skipping line, 'features' key not found: {line_str[:100]}...")
        return None

    flat_data = {}

    # Scalar features
    scalar_feature_keys = [
        "mean_words_per_comment", "median_words_per_comment", "mean_sents_per_comment",
        "median_sents_per_comment", "mean_words_per_sentence", "median_words_per_sentence",
        "sents_per_comment_skew", "words_per_sentence_skew", "total_double_whitespace",
        "punc_em_total", "punc_qm_total", "punc_period_total", "punc_comma_total",
        "punc_colon_total", "punc_semicolon_total", "flesch_reading_ease_agg",
        "gunning_fog_agg", "mean_word_len_overall", "ttr_overall", "mean_sentiment_neg",
        "mean_sentiment_neu", "mean_sentiment_pos", "mean_sentiment_compound",
        "std_sentiment_compound"
    ]
    for key in scalar_feature_keys:
        flat_data[key] = features_data.get(key)

    # q_scores (list of lists of floats)
    flat_data["q_scores"] = features_data.get("q_scores")

    # Tokenized data
    comments_tokenized = features_data.get("comments_tokenized", {}).get("data", {})
    if comments_tokenized: # Ensure it exists
        flat_data["input_ids"] = comments_tokenized.get("input_ids", {}).get("data")
        flat_data["token_type_ids"] = comments_tokenized.get("token_type_ids", {}).get("data")
        flat_data["attention_mask"] = comments_tokenized.get("attention_mask", {}).get("data")
    else: # If comments_tokenized or its 'data' sub-key is missing
        flat_data["input_ids"] = None
        flat_data["token_type_ids"] = None
        flat_data["attention_mask"] = None
        
    return flat_data

def define_arrow_schema():
    """
    Defines the PyArrow schema based on the expected data structure.
    This provides more robustness than inferring the schema.
    """
    fields = [
        # Scalar float features
        pa.field("mean_words_per_comment", pa.float64()),
        pa.field("median_words_per_comment", pa.float64()),
        pa.field("mean_sents_per_comment", pa.float64()),
        pa.field("median_sents_per_comment", pa.float64()),
        pa.field("mean_words_per_sentence", pa.float64()),
        pa.field("median_words_per_sentence", pa.float64()),
        pa.field("sents_per_comment_skew", pa.float64()),
        pa.field("words_per_sentence_skew", pa.float64()),
        pa.field("total_double_whitespace", pa.float64()), # Assuming float, could be int
        pa.field("punc_em_total", pa.float64()),
        pa.field("punc_qm_total", pa.float64()),
        pa.field("punc_period_total", pa.float64()),
        pa.field("punc_comma_total", pa.float64()),
        pa.field("punc_colon_total", pa.float64()),
        pa.field("punc_semicolon_total", pa.float64()),
        pa.field("flesch_reading_ease_agg", pa.float64()),
        pa.field("gunning_fog_agg", pa.float64()),
        pa.field("mean_word_len_overall", pa.float64()),
        pa.field("ttr_overall", pa.float64()),
        pa.field("mean_sentiment_neg", pa.float64()),
        pa.field("mean_sentiment_neu", pa.float64()),
        pa.field("mean_sentiment_pos", pa.float64()),
        pa.field("mean_sentiment_compound", pa.float64()),
        pa.field("std_sentiment_compound", pa.float64()),

        # List features
        pa.field("q_scores", pa.list_(pa.list_(pa.float64()))), # List of lists of floats

        # Tokenizer outputs (list of lists of ints)
        # The dtype was "torch.int64" in your example
        pa.field("input_ids", pa.list_(pa.list_(pa.int64()))),
        pa.field("token_type_ids", pa.list_(pa.list_(pa.int64()))),
        pa.field("attention_mask", pa.list_(pa.list_(pa.int64())))
    ]
    return pa.schema(fields)


def jsonl_to_arrow(jsonl_filepath, arrow_filepath):
    """
    Converts a JSONL file to an Arrow (Feather) file.
    """
    schema = define_arrow_schema()
    column_names = [field.name for field in schema]
    
    # Initialize lists to hold column data
    data_columns = {name: [] for name in column_names}

    print(f"Reading JSONL file: {jsonl_filepath}")
    with open(jsonl_filepath, 'r', encoding='utf-8') as f_jsonl:
        # Count lines for tqdm progress bar if file is large
        # For very large files, consider not doing this or using a more efficient way
        num_lines = sum(1 for _ in open(jsonl_filepath, 'r', encoding='utf-8'))
        f_jsonl.seek(0) # Reset file pointer

        for line in tqdm(f_jsonl, total=num_lines, desc="Processing lines"):
            flat_data_row = extract_data_from_json_line(line)
            if flat_data_row:
                for col_name in column_names:
                    data_columns[col_name].append(flat_data_row.get(col_name))
            # If flat_data_row is None, it means the line was skipped, so we don't append

    if not any(data_columns.values()): # Check if any data was actually processed
        print("No valid data processed. Arrow file will not be created.")
        return

    print("Creating PyArrow Table...")
    try:
        # Create PyArrow arrays for each column
        arrays = []
        for name in column_names:
            try:
                # Attempt to create array with specified type from schema
                # PyArrow can often infer, but explicit casting helps with Nones
                # For list types, ensure Nones are handled if items can be missing
                if data_columns[name] and isinstance(schema.field(name).type, (pa.ListType, pa.LargeListType)):
                     # For list types, we might need to be careful if individual rows have None for the whole list
                     # e.g., pa.array(data_columns[name], type=schema.field(name).type, safe=False) # if complex
                     arrays.append(pa.array(data_columns[name], type=schema.field(name).type))
                else:
                    arrays.append(pa.array(data_columns[name], type=schema.field(name).type))
            except pa.ArrowInvalid as e:
                print(f"Error creating array for column '{name}': {e}")
                print(f"First 5 problematic values: {data_columns[name][:5]}")
                # Fallback or raise error
                # For simplicity, let's try inferring type if explicit fails for some reason
                print(f"Attempting to infer type for column '{name}'...")
                arrays.append(pa.array(data_columns[name]))


        arrow_table = pa.Table.from_arrays(arrays, schema=schema)
    except Exception as e:
        print(f"Error creating PyArrow table: {e}")
        print("Trying to create table by inferring schema from pydict (might be less robust)...")
        # Fallback: let from_pydict infer (might not match desired complex types perfectly)
        arrow_table = pa.Table.from_pydict(data_columns)


    print(f"Writing Arrow (Feather) file: {arrow_filepath}")
    feather.write_feather(arrow_table, arrow_filepath)
    print("Conversion complete.")
    print(f"Arrow table schema:\n{arrow_table.schema}")

def main():
    parser = argparse.ArgumentParser(description="Convert JSONL file to Arrow (Feather) format.")
    parser.add_argument("jsonl_file", help="Path to the input JSONL file.")
    parser.add_argument("arrow_file", help="Path to the output Arrow (Feather) file.")
    args = parser.parse_args()

    jsonl_to_arrow(args.jsonl_file, args.arrow_file)

    # Example: Verify with Hugging Face datasets
    try:
        from datasets import load_from_disk, Features, Value, Sequence, ClassLabel, Dataset
        
        print(f"\nVerifying with Hugging Face datasets by loading '{args.arrow_file}'...")
        
        # For a single .arrow (Feather) file, you can load it like this:
        # Note: HF datasets `features` argument helps in being explicit
        # Construct HF Features object based on PyArrow schema
        hf_features_dict = {}
        pyarrow_schema = define_arrow_schema() # or loaded_table.schema
        for field in pyarrow_schema:
            if pa.types.is_float64(field.type) or pa.types.is_float32(field.type):
                hf_features_dict[field.name] = Value("float64") # or float32
            elif pa.types.is_int64(field.type) or pa.types.is_int32(field.type): # Add other int types if needed
                hf_features_dict[field.name] = Value("int64") # or int32
            elif pa.types.is_list(field.type) or pa.types.is_large_list(field.type):
                # This gets nested. For [[float]], it's Sequence(Sequence(Value("float64")))
                # For [[int]], it's Sequence(Sequence(Value("int64")))
                inner_value_type = field.type.value_type # This is pa.list_(<inner_type>)
                if pa.types.is_list(inner_value_type) or pa.types.is_large_list(inner_value_type):
                    innermost_value_type = inner_value_type.value_type
                    if pa.types.is_float64(innermost_value_type) or pa.types.is_float32(innermost_value_type):
                        hf_features_dict[field.name] = Sequence(Sequence(Value("float64")))
                    elif pa.types.is_int64(innermost_value_type) or pa.types.is_int32(innermost_value_type):
                         hf_features_dict[field.name] = Sequence(Sequence(Value("int64")))
                    else:
                        print(f"Warning: Unhandled innermost list type for HF Features: {innermost_value_type} for field {field.name}")
                        hf_features_dict[field.name] = Sequence(Sequence(Value("string"))) # Fallback
                else: # Simple list like list<int>
                    # This case is not in your example data for top-level features
                    print(f"Warning: Unhandled simple list type for HF Features: {inner_value_type} for field {field.name}")
                    hf_features_dict[field.name] = Sequence(Value("string")) # Fallback
            else:
                print(f"Warning: Unhandled PyArrow type for HF Features: {field.type} for field {field.name}")
                hf_features_dict[field.name] = Value("string") # Fallback

        hf_features = Features(hf_features_dict)
        
        # Load the Arrow file directly
        # dataset = Dataset.from_file(args.arrow_file, features=hf_features) # if it's Arrow IPC stream
        dataset = Dataset.from_pandas(feather.read_table(args.arrow_file).to_pandas(), features=hf_features)


        print("Successfully loaded into Hugging Face Dataset object.")
        print(dataset)
        if len(dataset) > 0:
            print("First record:", dataset[0])
        else:
            print("Dataset is empty.")

    except ImportError:
        print("\n`datasets` library not installed. Skipping Hugging Face verification.")
    except Exception as e:
        print(f"\nError during Hugging Face datasets verification: {e}")


if __name__ == "__main__":
    main()