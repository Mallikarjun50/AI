def vacuum_cleaner():
    # Initial environment: each room can be 'dirty' or 'clean'
    environment = {
        "A": "dirty",
        "B": "dirty"
    }

    # Initial location of the vacuum
    location = "A"

    # Run until both rooms are clean
    while environment["A"] == "dirty" or environment["B"] == "dirty":
        print(f"Vacuum is in room {location}")
        print(f"Room {location} is {environment[location]}")

        if environment[location] == "dirty":
            print(f"Cleaning room {location}...")
            environment[location] = "clean"
        else:
            print(f"Room {location} is already clean.")

        # Move to the other room
        location = "B" if location == "A" else "A"
        print("---")

    print("All rooms are clean!")

vacuum_cleaner()
