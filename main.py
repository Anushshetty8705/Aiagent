from graph import app

while True:
    query = input("\nEnter query: ")

    result = app.invoke({
        "query": query,
        "iteration": 0,
        "memory": []
    })

    print("\n========== FINAL ANSWER ==========\n")
    print(result["explanation"])

    print("\n========== TOP RESULT ==========\n")
    if result.get("ranked") is not None and len(result["ranked"]) > 0:
        print(result["ranked"][["model", "price", "score"]].head(3))