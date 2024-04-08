from json import dumps

import requests
from decouple import config
from langchain.tools import tool
from pydantic import BaseModel, Field


class WeatherInput(BaseModel):
    location: str = Field(description="Must be a valid location in city format.")


@tool("get_weather", args_schema=WeatherInput)
def get_weather(location: str) -> str:
    """Get the current weather for a specified location."""
    if not location:
        return (
            "Please provide a location and call the get_current_weather_function again."
        )
    API_params = {
        "key": config("WEATHER_API_KEY"),
        "q": location,
        "aqi": "no",
        "alerts": "no",
    }
    response: requests.models.Response = requests.get(
        "http://api.weatherapi.com/v1/current.json", params=API_params
    )
    str_response: str = dumps(response.json())
    return str_response


if __name__ == "__main__":
    print(get_weather.run("New York"))
