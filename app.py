import streamlit as st
from pipeline import run_research_pipeline

st.set_page_config(page_title="AI Research Agent", page_icon="🔍", layout="wide")

st.title("🔍 Multi-Agent Research System")
st.markdown("Powered by Mistral AI + Tavily + BeautifulSoup")

topic = st.text_input("Enter a research topic:", placeholder="e.g. Latest developments in quantum computing")

if st.button("Run Research", type="primary"):
    if not topic.strip():
        st.warning("Please enter a topic first.")
    else:
        with st.spinner("Step 1 — Search Agent is searching the web..."):
            search_agent = __import__('agents').build_search_agent()
            search_result = search_agent.invoke({
                "messages": [("user", f"Find recent, reliable, relevant and detailed information about: {topic}")]
            })
            search_results = search_result['messages'][-1].content

        st.subheader("🌐 Step 1 — Search Results")
        st.markdown(search_results)

        with st.spinner("Step 2 — Reader Agent is scraping the web..."):
            reader_agent = __import__('agents').build_reader_agent()
            reader_result = reader_agent.invoke({
                "messages": [("user",
                    f"Based on the following search results about '{topic}',"
                    f"Pick the most relevant URL and scrape it for deeper content.\n\n"
                    f"Search Results:\n{search_results[:800]}")]
            })
            scraped_content = reader_result['messages'][-1].content

        st.subheader("📄 Step 2 — Scraped Content")
        st.markdown(scraped_content)

        with st.spinner("Step 3 — Writer is drafting the report..."):
            from agents import writer_chain
            research_combined = (
                f"SEARCH RESULTS:\n{search_results}\n\n"
                f"DETAILED SCRAPED CONTENT:\n{scraped_content}\n"
            )
            report = writer_chain.invoke({"topic": topic, "research": research_combined})

        st.subheader("📝 Step 3 — Final Report")
        st.markdown(report)

        with st.spinner("Step 4 — Critic is reviewing the report..."):
            from agents import critic_chain
            feedback = critic_chain.invoke({"report": report})

        st.subheader("🧐 Step 4 — Critic Feedback")
        st.markdown(feedback)

        st.success("✅ Research complete!")