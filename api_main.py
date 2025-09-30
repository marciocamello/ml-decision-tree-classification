import asyncio
import uvicorn


async def main():
    config = uvicorn.Config(
        'api_modelo:app',
        host='0.0.0.0',
        port=8000,
        log_level='info',
        reload=True,
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == '__main__':
    asyncio.run(main())
