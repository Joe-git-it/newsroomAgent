import sys
from mcp.server.fastmcp import FastMCP
from newsroomagent.retrieval import retrieve
from datetime import datetime, timezone

# REGISTER SERVER
mcp = FastMCP("newsroomagent")

# ARCHIVE SEARCH TOOL TO CALL RETRIEVE FUNCTION
@mcp.tool()
def archive_search(query: str, k: int = 5) -> list[dict]:
    """Search the local news archive for past coverage on a topic.
    Returns up to `k` chunks with source filename and excerpt.
    Do not use for topics outside the archive or for recent events of
    the last 24 hours"""

    chunks = retrieve(query, k=k)
    return [
        {"source": c.metadata.get("source", "unknown"),
        "excerpt": c.page_content[:400]}
        for c in chunks
    ]

@mcp.tool()
def get_current_time() -> str:
    """Return the current UTC timestamp in ISO 8601 format. Use this when
    the user asks 'what time is it', when computing date math, or when
    the answer depends on what 'today' is."""
    return datetime.now(timezone.utc).isoformat()

if __name__ == "__main__":
    # STDERR ONLY
    print("STARTING NEWSROOMAGENT MCP SERVER", file=sys.stderr)  
    mcp.run(transport="stdio")


