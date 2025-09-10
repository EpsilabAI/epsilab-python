import os
import time
from datetime import datetime, timezone

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None

from epsilab import Epsilab


def main():
    if load_dotenv is not None:
        try:
            load_dotenv()
        except Exception:
            pass

    api_key = os.getenv("EPSILAB_API_KEY")
    api_base = os.getenv("EPSILAB_API_BASE", "https://www.epsilab.ai/api/ext/v1")

    if not api_key:
        print("EPSILAB_API_KEY not set. Create an API key in Account Settings and export it.")
        return

    client = Epsilab(api_base=api_base, api_key=api_key)

    start = time.time()
    iterations = 5  # 5 * 5s = 25s
    print(f"Polling live portfolio for ~{iterations * 5} seconds (every 5s)...\n")
    try:
        for i in range(iterations):
            try:
                ts = datetime.now(timezone.utc).isoformat()
                latest = client.get_live_latest(return_results_if_fresh=True)
                status = client.get_live_status()

                # Determine run id and timeframe
                run_id = latest.run_id or status.latest_run_id
                timeframe = status.timeframe or latest.timeframe

                # Fetch live data for this (or latest) run
                signals = client.get_portfolio_signals(run_id=run_id, limit=200)
                weights = client.get_portfolio_weights(run_id=run_id, limit=200)
                trades = client.get_live_trades(status="PENDING,EXECUTED,CLOSED", include_positions=False, limit=200)
                equity = client.get_live_equity(limit=200)
                members = client.get_live_members(run_id=run_id)

                next_eta = status.next_eta_minutes
                sig_rows = signals
                wt_rows = weights
                tr_rows = trades
                eq_series = equity
                mem_rows = members

                # Build samples for easy inspection
                def _sample_sig(r) -> str:
                    return r.log()

                def _sample_wt(r) -> str:
                    return r.log()

                def _sample_tr(r) -> str:
                    return r.log()

                def _sample_mem(r) -> str:
                    return r.log()

                eq_last = (eq_series[-1] if eq_series else None)
                eq_last_str = (eq_last.log() if eq_last else "n/a")

                print(f"[{ts}] runId={run_id} tf={timeframe} nextEtaMin={next_eta}")
                print(
                    f"  signals: {len(sig_rows)} | sample: " + ", ".join(map(_sample_sig, sig_rows[:2]))
                )
                print(
                    f"  weights: {len(wt_rows)} | sample: " + ", ".join(map(_sample_wt, wt_rows[:2]))
                )
                print(
                    f"  trades : {len(tr_rows)} | sample: " + ", ".join(map(_sample_tr, tr_rows[:2]))
                )
                print(
                    f"  members: {len(mem_rows)} | sample: " + ", ".join(map(_sample_mem, mem_rows[:2]))
                )
                print(
                    f"  equity : {len(eq_series)} pts | last: {eq_last_str}"
                )
            except Exception as e:
                print(f"Error during poll: {e}")

            # Wait ~5 seconds between polls
            try:
                time.sleep(5)
            except KeyboardInterrupt:
                print("Interrupted by user during sleep. Exiting...")
                break
    except KeyboardInterrupt:
        print("Interrupted by user. Exiting...")

    elapsed = time.time() - start
    print(f"\nDone. Elapsed ~{int(elapsed)}s.")


if __name__ == "__main__":
    main()


