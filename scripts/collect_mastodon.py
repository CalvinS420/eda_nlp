import requests
import pandas as pd
import time
import re

INSTANCE = "https://mastodon.social"

TARGETS = {
    "python": "technology",
    "photography": "photography",
    "music": "music"
}

TARGET_PER_CLASS = 60

headers = {
    "User-Agent": "COMP8240-research-project/1.0"
}

records = []

def get_next_max_id(link_header):
    if not link_header:
        return None

    match = re.search(r'max_id=([^&>]+).*?rel="next"', link_header)

    if match:
        return match.group(1)

    return None

for hashtag, label in TARGETS.items():

    print(f"\nCollecting #{hashtag} -> {label}")

    collected = 0
    max_id = None
    seen_ids = set()

    while collected < TARGET_PER_CLASS:

        url = f"{INSTANCE}/api/v1/timelines/tag/{hashtag}"

        params = {
            "limit": 40
        }

        if max_id:
            params["max_id"] = max_id

        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=30
        )

        if response.status_code != 200:
            print(
                f"HTTP error {response.status_code}: "
                f"{response.text[:200]}"
            )
            break

        posts = response.json()

        if not posts:
            print("No more posts returned.")
            break

        for post in posts:

            post_id = str(post["id"])

            if post_id in seen_ids:
                continue

            seen_ids.add(post_id)

            language = post.get("language")

            if language not in (None, "en"):
                continue

            records.append({
                "post_id": post_id,
                "source_hashtag": hashtag,
                "provisional_label": label,
                "created_at": post.get("created_at"),
                "language": language,
                "content_html": post.get("content", ""),
                "url": post.get("url"),
                "account": post.get("account", {}).get("acct", "")
            })

            collected += 1

            if collected >= TARGET_PER_CLASS:
                break

        print(f"Collected {collected}/{TARGET_PER_CLASS}")

        max_id = get_next_max_id(
            response.headers.get("Link")
        )

        if not max_id:
            break

        time.sleep(1)

df = pd.DataFrame(records)

df = df.drop_duplicates(subset="post_id")

output = "data/mastodon/mastodon_raw.csv"

df.to_csv(output, index=False)

print("\n====================================")
print("COLLECTION COMPLETE")
print("====================================")
print(f"Total unique posts: {len(df)}")
print("\nPosts by provisional label:")
print(df["provisional_label"].value_counts())
print(f"\nSaved to: {output}")
