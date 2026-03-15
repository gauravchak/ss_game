import os
import json
from dotenv import load_dotenv
from upstash_redis import Redis

# Load environment variables from .env file or rely on system env vars
load_dotenv()

def fetch_trajectories(output_file="trajectories.json"):
    url = os.environ.get("UPSTASH_REDIS_REST_URL")
    token = os.environ.get("UPSTASH_REDIS_REST_TOKEN")

    if not url or not token:
        print("Error: Required environment variables UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN are missing.")
        print("Please set them in a .env file or export them.")
        return

    print("Connecting to Upstash Redis...")
    redis = Redis(url=url, token=token)

    try:
        # Fetch all elements from the 'rl:trajectories' list
        # LRANGE rl:trajectories 0 -1 gets the whole list
        print("Fetching trajectories from list 'rl:trajectories'...")
        raw_trajectories = redis.lrange("rl:trajectories", 0, -1)
        
        if not raw_trajectories:
            print("No trajectories found in Redis. Play some games online first!")
            return

        print(f"Successfully fetched {len(raw_trajectories)} trajectories.")
        
        # Upstash python might return strings or dicts depending on JSON parsing
        parsed_trajectories = []
        for traj in raw_trajectories:
            if isinstance(traj, str):
                parsed_trajectories.append(json.loads(traj))
            else:
                parsed_trajectories.append(traj)

        with open(output_file, 'w') as f:
            json.dump(parsed_trajectories, f, indent=2)
            
        print(f"Saved {len(parsed_trajectories)} game trajectories to {output_file}")
        
    except Exception as e:
        print(f"Error fetching data: {e}")

if __name__ == "__main__":
    fetch_trajectories()
