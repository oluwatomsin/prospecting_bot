import streamlit as st
import pandas as pd
import asyncio
import json
from modules.jd_qualifier import job_classifier_function
from modules.company_info_extractor import get_company_info
from modules.lead_info_extractor import get_company_leads
import time

st.markdown("## Lead Classification Bot")

# Initialize all session states independently
for key in [
    "data_df", "classified_df", "filtered_df", "enriched_df", "edited_enriched_df", "final_classified_df"
]:
    if key not in st.session_state:
        st.session_state[key] = None

# Upload CSV
if data := st.file_uploader("Kindly Upload your csv file containing the Job post and company info", type=['csv']):
    st.write("File uploaded successfully")
    df = pd.read_csv(data).copy(deep=True)
    st.session_state.data_df = df
    st.data_editor(df)

    job_post_col = st.selectbox(
        "Select the column containing the job post text:",
        df.columns,
        index=0,
        key="job_post_col"
    )

    if st.button("Classify Job Posts"):
        st.info("Scoring job posts. This may take a while for large files...")

        async def classify_all_jobs():
            tasks = [
                job_classifier_function(str(row[job_post_col]))
                for _, row in st.session_state.data_df.iterrows()
            ]
            results = await asyncio.gather(*tasks)

            classes, reasons = [], []
            for result in results:
                try:
                    parsed = json.loads(result)
                    classes.append(parsed.get("label", ""))
                    reasons.append(parsed.get("reason", ""))
                except Exception:
                    classes.append("")
                    reasons.append(result)

            classified_df = st.session_state.data_df.copy(deep=True)
            classified_df["class"] = classes
            classified_df["reason"] = reasons
            return classified_df

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        classified = loop.run_until_complete(classify_all_jobs())
        st.session_state.classified_df = classified.copy(deep=True)
        st.success("Classification complete!")

# Display classified data
if st.session_state.classified_df is not None:
    st.markdown("### Classified Job Posts")
    st.dataframe(st.session_state.classified_df)

    if st.button("Remove disqualified jobs"):
        filtered_df = st.session_state.classified_df[
            st.session_state.classified_df["class"] != "Disqualified"
        ].reset_index(drop=True).copy(deep=True)
        st.session_state.filtered_df = filtered_df
        st.info(f"{len(filtered_df)} jobs remain after filtering.")

# Show filtered data
if st.session_state.filtered_df is not None:
    st.markdown("### Data After Removing Disqualified Jobs")
    st.dataframe(st.session_state.filtered_df)

    company_col = st.selectbox(
        "Select the column containing the company name:",
        st.session_state.filtered_df.columns,
        index=0,
        key="company_col"
    )

    if st.button("Fetch Company Website & LinkedIn"):
        st.info("Fetching company info...")

        async def fetch_company_info_all():
            tasks = [
                asyncio.to_thread(get_company_info, str(row[company_col]))
                for _, row in st.session_state.filtered_df.iterrows()
            ]
            results = await asyncio.gather(*tasks)
            websites = [r.get("website", "") for r in results]
            linkedins = [r.get("linkedin", "") for r in results]

            enriched_df = st.session_state.filtered_df.copy(deep=True)
            enriched_df["Company_website"] = websites
            enriched_df["Company_linkedIn"] = linkedins

            if "website_content" not in enriched_df.columns:
                enriched_df["website_content"] = ""
            if "linked_content" not in enriched_df.columns:
                enriched_df["linked_content"] = ""

            return enriched_df

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        enriched_df = loop.run_until_complete(fetch_company_info_all())
        st.session_state.enriched_df = enriched_df.copy(deep=True)
        st.success("Company info fetched.")

# Show editable enriched data
if st.session_state.enriched_df is not None:
    st.markdown("### Data With Company Website & LinkedIn (Editable)")
    edited_df = st.data_editor(st.session_state.enriched_df, key="editable_company_data")

    if st.button("Save Edits to Company Info"):
        st.session_state.edited_enriched_df = edited_df.copy(deep=True)
        st.success("Company data edits saved.")

# --- Final Company Classification ---
if (
    st.session_state.edited_enriched_df is not None
    and "website_content" in st.session_state.edited_enriched_df.columns
    and "linked_content" in st.session_state.edited_enriched_df.columns
):
    st.markdown("### Company Qualification")

    if st.button("Classify Companies"):
        st.info("Classifying companies...")

        async def classify_companies_all():
            from modules.company_info_qualifier import company_classifier_function
            tasks = [
                company_classifier_function(
                    str(row["linked_content"]),
                    str(row["website_content"])
                )
                for _, row in st.session_state.edited_enriched_df.iterrows()
            ]
            results = await asyncio.gather(*tasks)

            labels, reasons = [], []
            for res in results:
                try:
                    parsed = json.loads(res)
                    labels.append(parsed.get("label", ""))
                    reasons.append(parsed.get("reason", ""))
                except Exception as e:
                    labels.append("")
                    reasons.append(f"Error: {str(e)}")

            final_df = st.session_state.edited_enriched_df.copy(deep=True)
            final_df["company_class"] = labels
            final_df["company_class_reason"] = reasons
            return final_df

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        final_classified = loop.run_until_complete(classify_companies_all())
        st.session_state.final_classified_df = final_classified.copy(deep=True)
        st.success("Company classification complete!")


# --- Show & Filter Final Company Classification ---
if (
    st.session_state.final_classified_df is not None
    and "company_class" in st.session_state.final_classified_df.columns
):
    st.markdown("### Final Data With Company Class")
    st.data_editor(st.session_state.final_classified_df, key="final_company_class_editor")

    if st.button("Remove Disqualified Companies"):
        qualified_df = st.session_state.final_classified_df[
            st.session_state.final_classified_df["company_class"] != "Disqualified"
        ].reset_index(drop=True).copy(deep=True)
        st.session_state.qualified_companies_df = qualified_df
        st.success(f"{len(qualified_df)} companies remain after filtering Disqualified ones.")

        st.markdown("### Qualified Companies Only")
        st.data_editor(qualified_df, key="qualified_company_class_editor")


if "qualified_companies_df" in st.session_state:

    if st.button("🔍 Generate Leads for Each Company"):
        st.info("Generating leads... this might take a while ⏳")

        updated_df = st.session_state.qualified_companies_df.copy()

        leads_results = []
        for index, row in updated_df.iterrows():
            company_name = row["companyName"]

            try:
                leads = get_company_leads(company_name)
                leads_results.append(json.dumps(leads, ensure_ascii=False))
            except Exception as e:
                st.warning(f"❌ Failed to generate leads for {company_name}: {e}")
                leads_results.append("[]")  # Empty JSON array fallback


        updated_df["leads"] = leads_results
        st.session_state.leads_df = updated_df

        st.success("✅ Lead generation complete.")
        st.dataframe(updated_df)

    elif "leads_df" in st.session_state:
        st.markdown("### ✅ Leads Data")
        st.dataframe(st.session_state.leads_df)

# --- Allow user to select columns and download as CSV ---
if "leads_df" in st.session_state and st.session_state.leads_df is not None:
    st.markdown("### 📥 Download Final Data")

    all_columns = list(st.session_state.leads_df.columns)
    selected_columns = st.multiselect(
        "Select columns to include in the downloaded CSV:",
        all_columns,
        default=all_columns  # preselect all by default
    )

    if selected_columns:
        filtered_download_df = st.session_state.leads_df[selected_columns]
        csv_data = filtered_download_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📩 Download CSV",
            data=csv_data,
            file_name="qualified_leads.csv",
            mime="text/csv"
        )
    else:
        st.warning("⚠️ Please select at least one column to download.")
