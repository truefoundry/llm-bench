from asyncio.tasks import Task
import argparse
import asyncio
import numpy as np
import aiohttp
import time
import collections
import contextlib
import math
import functools


class MetricsCollector:
    def __init__(self, user_def, session_time=None, ping_latency=0.0):
        self.start_time = math.floor(time.time())
        self.input_word_bucket = collections.defaultdict(int)
        self.response_word_bucket = collections.defaultdict(int)
        self.response_head_latency_bucket = collections.defaultdict(list)
        self.response_latency_bucket = collections.defaultdict(list)
        self.on_going_requests = 0
        self.response_bucket = collections.defaultdict(int)
        self.total_requests = 0
        self.on_going_users = 0
        self.status_bucket = collections.defaultdict(int)
        self.user_def = user_def
        self.session_time = session_time
        self.ping_latency = ping_latency

    def collect_response_chunk(self, chunk: list, input_tokens: int = 0):
        self.response_word_bucket[math.floor(time.time())] += len(chunk)
        self.input_word_bucket[math.floor(time.time())] += input_tokens

    def collect_response_status(self, status):
        self.status_bucket[status] += 1

    def collect_response_head_latency(self, latency):
        self.response_head_latency_bucket[math.floor(time.time())] += [
            latency - self.ping_latency
        ]

    @contextlib.contextmanager
    def collect_http_request(self):
        start_time = time.time()
        self.on_going_requests += 1
        yield
        self.on_going_requests -= 1
        self.response_bucket[math.floor(time.time())] += 1
        self.response_latency_bucket[math.floor(time.time())] += [
            time.time() - start_time - self.ping_latency
        ]

    @contextlib.contextmanager
    def collect_user(self):
        self.on_going_users += 1
        yield
        self.on_going_users -= 1

    async def report_loop(self, time_window=5):
        """
        Each bucket is in 1s. This function will report the avg metrics in the past time_window seconds.
        """
        while True:
            await asyncio.sleep(time_window)
            now = math.floor(time.time())
            print(f"Time: {now - self.start_time}")
            print(f"Active Users: {self.on_going_users}")
            print(
                f"Request/s: {sum(self.response_bucket[i] for i in range(now - time_window, now)) / time_window}"
            )
            print(f"Total Requests: {self.total_requests}")
            print(f"Active Requests: {self.on_going_requests}")
            latency_bucket = [
                j
                for i in range(now - time_window, now)
                for j in self.response_head_latency_bucket[i]
            ]
            if latency_bucket:
                print(f"Response Head Latency: {np.mean(latency_bucket)}")
            latency_bucket = [
                j
                for i in range(now - time_window, now)
                for j in self.response_latency_bucket[i]
            ]
            if latency_bucket:
                print(f"Response Latency: {np.mean(latency_bucket)}")
            print(
                f"Response Tokens/s: {sum(self.response_word_bucket[i] for i in range(now - time_window, now)) / time_window}"
            )
            print(
                f"Input Tokens/s: {sum(self.input_word_bucket[i] for i in range(now - time_window, now)) / time_window}"
            )
            print(f"Status: {self.status_bucket}")
            print()

            if self.session_time and now - self.start_time >= self.session_time:
                self.report_final()
                break

    def report_final(self):
        print("=================== Final Report ====================")
        print(f"Total Requests: {self.total_requests}")
        print(
            f"Average Request/s: {self.total_requests / (time.time() - self.start_time)}"
        )

        head_latency_size = sum(len(i) for i in self.response_head_latency_bucket.values())
        if head_latency_size:
            head_latencies = [j for i in self.response_head_latency_bucket.values() for j in i]

            print(
                f"Average Response Head Latency: {sum(head_latencies) / head_latency_size}"
            )
            print(
                f"Median Response Head Latency: {np.percentile(head_latencies, 50)}"
            )
            print(
                f"95% Response Head Latency: {np.percentile(head_latencies, 95)}"
            )
            print(
                f"99% Response Head Latency: {np.percentile(head_latencies, 99)}"
            )

        latency_size = sum(len(i) for i in self.response_latency_bucket.values())
        if latency_size:
            latencies = [j for i in self.response_latency_bucket.values() for j in i]
            print(
                f"Average Response Latency: {sum(latencies) / latency_size}"
            )
            print(
                f"Median Response Latency: {np.percentile(latencies, 50)}"
            )
            print(
                f"95% Response Latency: {np.percentile(latencies, 95)}"
            )
            print(
                f"99% Response Latency: {np.percentile(latencies, 99)}"
            )

        print(
            f"Average Response Tokens/s: {sum(self.response_word_bucket.values()) / (time.time() - self.start_time)}"
        )
        print(
            f"Average Input Tokens/s: {sum(self.input_word_bucket.values()) / (time.time() - self.start_time)}"
        )


def linear_regression(x, y):
    x = tuple((i, 1) for i in x)
    y = tuple(i for i in y)
    a, b = np.linalg.lstsq(x, y, rcond=None)[0]
    return a, b


class UserSpawner:
    def __init__(
        self,
        user_def,
        collector: MetricsCollector,
        target_user_count=None,
        target_time=None,
    ):
        self.target_user_count = 1 if target_user_count is None else target_user_count
        self.target_time = time.time() + 10 if target_time is None else target_time

        self.data_collector = collector
        self.user_def = user_def

        self.user_list: list[Task] = []

    async def sync(self):
        while True:
            if self.current_user_count == self.target_user_count:
                return
            await asyncio.sleep(0.1)

    @property
    def current_user_count(self):
        return len(self.user_list)

    async def user_loop(self):
        with self.data_collector.collect_user():
            cookie_jar = aiohttp.DummyCookieJar()
            try:
                async with aiohttp.ClientSession(cookie_jar=cookie_jar) as session:
                    while True:
                        url, headers, data, input_data = self.user_def.make_request()
                        self.data_collector.total_requests += 1
                        with self.data_collector.collect_http_request():
                            req_start = time.time()
                            async with session.post(
                                url,
                                headers=headers,
                                data=data,
                            ) as response:
                                self.data_collector.collect_response_status(
                                    response.status
                                )
                                try:
                                    if response.status != 200:
                                        continue

                                    first = True
                                    async for data, end_of_http_chunk in response.content.iter_chunks():
                                        result = self.user_def.parse_response(data)
                                        if first:
                                            first = False
                                            self.data_collector.collect_response_head_latency(
                                                time.time() - req_start
                                            )
                                        self.data_collector.collect_response_chunk(
                                            result, input_data['input_tokens']
                                        )
                                        if not end_of_http_chunk:
                                            break
                                except Exception as e:
                                    self.data_collector.collect_response_status(str(e))
                                    raise e
                        await self.user_def.rest()
            except asyncio.CancelledError:
                pass

    def spawn_user(self):
        self.user_list.append(asyncio.create_task(self.user_loop()))

    async def cancel_all_users(self):
        try:
            user = self.user_list.pop()
            user.cancel()
        except IndexError:
            pass
        await asyncio.sleep(0)

    async def spawner_loop(self):
        while True:
            current_users = len(self.user_list)
            if current_users == self.target_user_count:
                await asyncio.sleep(0.1)
            elif current_users < self.target_user_count:
                self.spawn_user()
                sleep_time = max(
                    (self.target_time - time.time())
                    / (self.target_user_count - current_users),
                    0,
                )
                await asyncio.sleep(sleep_time)
            elif current_users > self.target_user_count:
                self.user_list.pop().cancel()
                sleep_time = max(
                    (time.time() - self.target_time)
                    / (current_users - self.target_user_count),
                    0,
                )
                await asyncio.sleep(sleep_time)

    async def aimd_loop(
        self,
        adjust_interval=5,
        sampling_interval=5,
        ss_delta=1,
    ):
        """
        Detect a suitable number of users to maximize the words/s.
        """
        while True:
            while True:
                # slow start
                now = math.floor(time.time())
                words_per_seconds = [
                    self.data_collector.response_word_bucket[i]
                    for i in range(now - sampling_interval, now)
                ]
                slope = linear_regression(
                    range(len(words_per_seconds)), words_per_seconds
                )[0]
                if slope >= -0.01:
                    # throughput is increasing
                    cwnd = self.current_user_count
                    target_cwnd = max(int(cwnd * (1 + ss_delta)), cwnd + 1)
                    self.target_user_count = target_cwnd
                    self.target_time = time.time() + adjust_interval
                    print(f"SS: {cwnd} -> {target_cwnd}")
                    await asyncio.sleep(adjust_interval)
                else:
                    # throughput is decreasing, stop slow start
                    cwnd = self.current_user_count
                    target_cwnd = math.ceil(cwnd * 0.5)
                    self.target_user_count = target_cwnd
                    self.target_time = time.time() + adjust_interval
                    print(f"SS Ended: {target_cwnd}")
                    break

            await self.sync()
            await asyncio.sleep(min(adjust_interval, sampling_interval, 10))
            return 0


async def start_benchmark_session(args, user_def):
    # ping server
    response_times = []
    async with aiohttp.ClientSession() as session:
        async with session.get(user_def.ping_url()) as response:
            print(response.status)
            assert response.status in [200, 404]
        await asyncio.sleep(0.3)

        for _ in range(5):
            time_start = time.time()
            async with session.get(user_def.ping_url()) as response:
                assert response.status in [200, 404]
            response_times.append(time.time() - time_start)
            await asyncio.sleep(0.3)
    ping_latency = sum(response_times) / len(response_times)
    print(f"Ping latency: {ping_latency}. ping correction: {args.ping_correction}")

    # init
    collector = MetricsCollector(
        user_def, args.session_time, ping_latency - 0.005 if args.ping_correction else 0
    )
    user_spawner = UserSpawner(
        user_def, collector, args.max_users, target_time=time.time() + 20
    )
    asyncio.create_task(user_spawner.spawner_loop())
    asyncio.create_task(collector.report_loop())
    if args.max_users is None:
        asyncio.create_task(user_spawner.aimd_loop())

    if args.session_time is not None:
        await asyncio.sleep(args.session_time + 1)
    else:
        await asyncio.wait(user_spawner.user_list)

    await user_spawner.cancel_all_users()
    return 0