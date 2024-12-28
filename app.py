from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from phi.agent import Agent, RunResponse
from phi.model.openai import OpenAIChat
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from dotenv import load_dotenv
import os
import openai
load_dotenv()


# Initialize FastAPI app
app = FastAPI()


openai.api_key=os.getenv("OPENAI_API_KEY")


@app.post("/ask")
async def ask_question(question: str = Form(...)):
        try:
            web_search_agent = Agent(
                name="web search agent",
                role="search the web for information",
                model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
                # model=OpenAIChat(id="gpt-4o"),
                tools=[DuckDuckGo()],
                instructions=["Always include sources"],
                show_tool_calls=True,
                markdown=True,
            )
            finance_agent = Agent(
                name="Finance AI Agent",
                model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
                # model=OpenAIChat(id="gpt-4o"),
                tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)],
                description="You are an investment analyst that researches stock prices, analyst recommendations, and stock fundamentals.",
                instructions=["Format your response using markdown and use tables to display data where possible."],
                show_tool_calls=True,
                markdown=True,
            )
            ## combining all agents
            multi_ai_agent=Agent(
                team=[web_search_agent,finance_agent],
                instructions=["Always include sources, and use tables to display data where possible."],
                show_tool_calls=True,
                markdown=True,
            )

            response: RunResponse = multi_ai_agent.run(question)
            finalresponse = response.content

        
            # response= multi_ai_agent.print_response(question, stream=True)
            return JSONResponse(content={"response": finalresponse}, status_code=200)

        except Exception as e:
             raise HTTPException(status_code=500, detail=str(e))



# web_search_agent.print_response("Tell me about OpenAI Sora?", stream=True)
# response= multi_ai_agent.print_response("summarize analyst recommendation and share the latest news for Meta", stream=True)
# print(response)

# print(os.getenv("GROQ_API_KEY"))