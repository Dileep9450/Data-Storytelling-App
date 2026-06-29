import streamlit as st
import pandas as pd

# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="Titanic Data Storytelling App",
    layout="wide"
)

st.title("🚢 Titanic Data Storytelling App")
st.markdown(
    "Explore the Titanic dataset through data storytelling, visualizations, and insights."
)

# =====================================
# FILE UPLOAD
# =====================================

uploaded_file = st.file_uploader(
    "📂 Upload Titanic CSV Dataset",
    type=["csv"]
)

# =====================================
# MAIN APP
# =====================================

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        # ---------------------------------
        # CLEAN COLUMN NAMES
        # ---------------------------------

        df.columns = (
            df.columns
            .astype(str)
            .str.lower()
            .str.strip()
        )

        # ---------------------------------
        # HANDLE MISSING VALUES
        # ---------------------------------

        if "age" in df.columns:
            df["age"] = pd.to_numeric(
                df["age"],
                errors="coerce"
            )
            df["age"] = df["age"].fillna(
                df["age"].median()
            )

        if "fare" in df.columns:
            df["fare"] = pd.to_numeric(
                df["fare"],
                errors="coerce"
            )
            df["fare"] = df["fare"].fillna(
                df["fare"].median()
            )

        # Convert object columns to string
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype(str)

        # =====================================
        # DATASET INTRODUCTION
        # =====================================

        st.header("📖 Dataset Introduction")

        st.write("""
        The Titanic dataset contains information about passengers
        aboard the RMS Titanic including survival status, gender,
        age, passenger class, fare paid, and embarkation details.
        """)

        st.subheader("Dataset Preview")
        st.dataframe(
            df.head(),
            use_container_width=True
        )

        # =====================================
        # DATASET OVERVIEW
        # =====================================

        st.header("📊 Dataset Overview")

        c1, c2, c3 = st.columns(3)

        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing Values", int(df.isnull().sum().sum()))

        # =====================================
        # EDA
        # =====================================

        st.header("🔍 Exploratory Data Analysis (EDA)")

        st.subheader("Data Types")

        dtype_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": [str(x) for x in df.dtypes]
        })

        st.dataframe(
            dtype_df,
            use_container_width=True
        )

        st.subheader("Missing Values")

        missing_df = pd.DataFrame({
            "Column": df.columns,
            "Missing Values": df.isnull().sum().values
        })

        st.dataframe(
            missing_df,
            use_container_width=True
        )

        st.subheader("Summary Statistics")

        numeric_cols = df.select_dtypes(
            include="number"
        )

        if not numeric_cols.empty:
            st.dataframe(
                numeric_cols.describe(),
                use_container_width=True
            )

        # =====================================
        # VISUALIZATIONS
        # =====================================

        st.header("📈 Visualizations")

        # 1
        if "survived" in df.columns:

            st.subheader("1️⃣ Survival Distribution")

            survival = (
                df["survived"]
                .replace({
                    0: "Not Survived",
                    1: "Survived"
                })
                .value_counts()
            )

            st.bar_chart(survival)

        # 2
        if "sex" in df.columns:

            st.subheader("2️⃣ Gender Distribution")

            st.bar_chart(
                df["sex"].value_counts()
            )

        # 3
        if "pclass" in df.columns:

            st.subheader("3️⃣ Passenger Class Distribution")

            st.bar_chart(
                df["pclass"].value_counts()
            )

        # 4
        if (
            "fare" in df.columns
            and "pclass" in df.columns
        ):

            st.subheader("4️⃣ Average Fare by Class")

            fare_class = (
                df.groupby("pclass")["fare"]
                .mean()
            )

            st.bar_chart(fare_class)

        # 5
        if "embarked" in df.columns:

            st.subheader("5️⃣ Embarkation Port Distribution")

            st.bar_chart(
                df["embarked"]
                .value_counts()
            )

        # 6
        if "age" in df.columns:

            st.subheader("6️⃣ Age Distribution")

            age_sorted = (
                df["age"]
                .sort_values()
                .reset_index(drop=True)
            )

            st.line_chart(age_sorted)

        # =====================================
        # INSIGHTS
        # =====================================

        st.header("💡 Insights and Findings")

        insights = []

        if "survived" in df.columns:

            survival_rate = round(
                df["survived"].mean() * 100,
                2
            )

            insights.append(
                f"Overall survival rate: {survival_rate}%"
            )

        if "sex" in df.columns:

            top_gender = (
                df["sex"]
                .value_counts()
                .idxmax()
            )

            insights.append(
                f"Most passengers were {top_gender}."
            )

        if "pclass" in df.columns:

            common_class = (
                df["pclass"]
                .value_counts()
                .idxmax()
            )

            insights.append(
                f"Most passengers travelled in Class {common_class}."
            )

        for item in insights:
            st.write("✅", item)

        # =====================================
        # CONCLUSION
        # =====================================

        st.header("📌 Final Conclusion / Recommendations")

        st.success("""
        Analysis of the Titanic dataset suggests that
        passenger class, age, gender, and fare played
        important roles in survival outcomes.

        Recommendations:
        • Prioritize vulnerable passengers during emergencies.
        • Improve safety planning through data analysis.
        • Use predictive analytics for transportation safety.
        """)

    except Exception as e:

        st.error(f"Error reading dataset: {e}")

else:

    st.info(
        "📂 Upload a Titanic CSV file to begin storytelling."
    )