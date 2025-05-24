import ray
import math
import time
import random

ray.init()

"""
follow tutorial here:
https://docs.ray.io/en/latest/ray-core/examples/monte_carlo_pi.html
"""

@ray.remote
class ProgressActor:
	def __init__(self, total_num_samples: int):
		self.total_num_samples = total_num_samples
		self.num_samples_completed_per_task = {}

	def report_progress(self, task_id: int, num_samples_completed: int) -> None:
		self.num_samples_completed_per_task[task_id] = num_samples_completed

	def get_progress(self) -> float:
		return(
			sum(self.num_samples_completed_per_task.values()) / self.total_num_samples
		)


@ray.remote
def sampling_task(num_samples: int, task_id: int,
					progress_actor: ray.actor.ActorHandle) -> int:
	
	"""
	this shows an example of calling actor methods from tasks
	"""
	num_inside = 0
	for i in range(num_samples):
		x, y = random.uniform(-1, 1), random.uniform(-1, 1)
		if math.hypot(x, y) <= 1:
			num_inside += 1

		if (i + 1) % 1_000_000 == 0:
			progress_actor.report_progress.remote(task_id, i + 1) # this is async

	#report final progress
	progress_actor.report_progress.remote(task_id, num_samples)
	return num_samples



if __name__ == "__main__":
	NUM_SAMPLING_TASKS = 5
	NUM_SAMPLES_PER_TASK = 10_000_000
	TOTAL_NUM_SAMPLES = NUM_SAMPLING_TASKS * NUM_SAMPLES_PER_TASK

	# this creates and runs the actor on a remote worker process
	progress_actor = ProgressActor.remote(TOTAL_NUM_SAMPLES)

	# create and execute all sampling tasks in parallel
	results = [
		sampling_task.remote(NUM_SAMPLES_PER_TASK, i, progress_actor)
		for i in range(NUM_SAMPLING_TASKS)
	]

	while True:
		progress = ray.get(progress_actor.get_progress.remote())
		print(f"Progress: {int(progress * 100)}%")

		if progress == 1:
			break

		time.sleep(1)

	total_num_inside = sum(ray.get(results))
	pi = (total_num_inside * 4) / TOTAL_NUM_SAMPLES
	print(f"Estimated value of pi is: {pi}")
