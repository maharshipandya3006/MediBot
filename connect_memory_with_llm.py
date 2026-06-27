from app import create_rag_chain, format_page_label, format_source_name


def main():
    user_query = input("Write Query Here: ").strip()

    if not user_query:
        print("Please enter a valid question.")
        return

    rag_chain = create_rag_chain()
    response = rag_chain.invoke({"input": user_query})

    print("\nRESULT:")
    print(response.get("answer", "I do not know based on the available context."))

    print("\nSOURCE DOCUMENTS:")
    for doc in response.get("context", []):
        metadata = doc.metadata or {}
        source = format_source_name(metadata.get("source"))
        page = format_page_label(metadata)
        preview = doc.page_content[:200].replace("\n", " ")

        print(f"- Source: {source}, Page: {page}")
        print(f"  Preview: {preview}")


if __name__ == "__main__":
    main()
