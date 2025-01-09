import streamlit as st
import plotly.express as px
from main import load_data, group_and_aggregate_data, remove_sparse_columns, dimensionality_reduction

# Streamlit UI
st.title("Knesset Election Data Visualization")
st.sidebar.header("Settings")

# File Upload
uploaded_file = st.sidebar.file_uploader("Upload your dataset (.csv, .xls, .xlsx)", type=["csv", "xls", "xlsx"])

if uploaded_file:
    # Save the uploaded file to a temporary file
    temp_file_path = f"uploaded_file_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        # Load the data using the load_data function
        df = load_data(temp_file_path)
        st.write("Dataset Loaded Successfully!")
        st.dataframe(df)
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        st.stop()

    num_components = st.sidebar.slider("Number of PCA components:", min_value=1, max_value=10, value=2)

    # Processing mode
    process_mode = st.sidebar.radio("Select processing mode:", ["City-wise", "Party-wise"], help="In Party-wise, after transposing the data, each row represents a single party, and each column represents a city. Since the dataset no longer has multiple rows for the same category (e.g., cities), group-by operations are not applicable.")

    if process_mode == "City-wise":
        # Existing City-wise logic (unchanged)
        try:
            group_by_column = st.sidebar.selectbox("Select column to group by:", df.columns)
            agg_func = st.sidebar.selectbox("Select aggregation function:", ["sum", "mean", "count"])
            threshold = st.sidebar.number_input("Threshold for sparse column removal:", min_value=0, value=1000)

            grouped_data = group_and_aggregate_data(df, group_by_column, agg_func)
            filtered_data = remove_sparse_columns(grouped_data, threshold)

            reduced_data = dimensionality_reduction(
                filtered_data.reset_index(), num_components=num_components, meta_columns=[group_by_column]
            )

            st.write("Reduced Data:")
            st.dataframe(reduced_data)

            fig = px.scatter(
                reduced_data,
                x="PC1",
                y="PC2",
                hover_name=group_by_column,
                title="Dimensionality Reduction Visualization (City-wise)"
            )
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"City-wise processing failed: {e}")



    elif process_mode == "Party-wise":
        # Group and aggregate data
        grouped_data = group_and_aggregate_data(df, 'city_name', sum)

        # Remove sparse columns
        threshold = st.sidebar.number_input("Threshold for sparse column removal:", min_value=0, value=1000)
        try:
            filtered_parties = remove_sparse_columns(grouped_data, threshold)

            # Transpose data for party comparison
            transposed_data = filtered_parties.T

            # Filter cities with total votes >= 1000
            filtered_cities = remove_sparse_columns(transposed_data, threshold)

            # Perform dimensionality reduction manually for parties using the provided function
            reduced_parties = dimensionality_reduction(
                filtered_cities.reset_index(), num_components=num_components, meta_columns=['index']
            )

            # Add party names as a new column to be used for hover
            reduced_parties['party_name'] = reduced_parties['index']

            # Display results
            st.write("Reduced Data:")
            st.dataframe(reduced_parties)

            # Scatter plot for parties
            fig_parties = px.scatter(
                reduced_parties,
                x='PC1',
                y='PC2',
                hover_name='party_name',
                title="Dimensionality Reduction for Parties (num_components=2)"
            )
            st.plotly_chart(fig_parties)

        except Exception as e:

            st.error(f"Party-wise processing failed: {e}")

    # Cleanup
    try:
        with open(temp_file_path, "w") as f:
            pass
    except Exception:
        pass
