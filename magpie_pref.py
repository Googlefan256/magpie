from typing import Optional
from lib.api import magpie_preference
import concurrent.futures
import json


def do(
    times: int,
    workers: int,
    system: Optional[str],
    out: str,
):
    def task_function():
        return magpie_preference(system)

    tasks = [task_function for _ in range(times)]
    res = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for i in range(0, times, workers):
            for task in tasks[i : i + workers]:
                future = executor.submit(task)
                futures.append(future)
        for future in concurrent.futures.as_completed(futures):
            try:
                futr = future.result()
                if futr is not None:
                    res.append(futr)
                    print("Task done")
                    json.dump(res, open(out, "w"), ensure_ascii=False)
                else:
                    print("Task failed")
            except Exception as e:
                print(f"An error occurred: {e}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--times", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=50)
    parser.add_argument("--system", type=str)
    parser.add_argument("--out", type=str, default="out.json")
    args = parser.parse_args()
    do(args.times, args.workers, args.system, args.out)
