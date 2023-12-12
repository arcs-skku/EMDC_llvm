#ifndef BEMPS_HPP
#define BEMPS_HPP

#include <semaphore.h>

#define BEMPS_STATS

#define BEMPS_BEACON_BUF_SZ 0x1000

#define BEMPS_SCHED_TIMEOUT_NS 100000000UL  // 100ms
//#define BEMPS_SCHED_TIMEOUT_NS 5000000000UL // 5s
//#define BEMPS_SCHED_TIMEOUT_NS 10000000000UL // 10s

// The beacon type for an application calling the bemps library.
typedef struct {
  long mem_B; // Signed. Negatives currently used for frees. Improve later.
  long warps; // same
  long thread_blocks; // same
  long actual_mem_B; // add for actual mem_req of extra task

  bool changed; // 1 = eager free
  // TODO: constant memory
} bemps_beacon_t;

typedef struct {
  sem_t sem;  // for signaling the process to launch its kernel
  int device_id;
} bemps_sched_notif_t;

typedef enum {
  BEMPS_BEACON_STATE_COMPLETED_E = 0,
  BEMPS_BEACON_STATE_BEACON_FIRED_E,
  BEMPS_BEACON_STATE_SCHEDULER_READ_E,
  BEMPS_BEACON_STATE_SCHEDULED_E
} bemps_beacon_state_e;

typedef struct {
  long long timestamp_ns;
  pid_t pid;
  int age;
  int exit_flag;
  bemps_beacon_state_e state;
  bemps_beacon_t beacon;
  bemps_sched_notif_t sched_notif;
  bool extra_status; // for checking kernel end, 1 = extra_task
  size_t extra_mem; // mem which task can achieve
} bemps_shm_comm_t;

// A single type that captures all shared memory pointers between the bemps
// library and the scheduler
typedef struct {

  // XXX Do not reorder this struct without taking into account
  // jobs_running_on_gpu and jobs_waiting_on_gpu, and how the workloader gets
  // their offset inside of read_shm()

  // The head index into the circuluar buffer of beacons
  int beacon_q_head;

  // The tail index into the circuluar buffer of beacons
  int beacon_q_tail;

  // For batching algorithms, the size of the batch before we signal the
  // scheduler.
  int max_batch_size;

  // For jobs with dynamic job pressure, these variables can be read by a
  // driver to help decide whether or not increase or decrease the number of
  // running jobs. There's no lock. Drivers can re-read these values if they
  // seem out-of-sync with their own internal numbers.
  // XXX See comment at the top of this struct regarding workloader's
  // read_shm() function and the offset of these struct members.
  int jobs_running_on_gpu;
  int jobs_waiting_on_gpu;

  // Lock and cond var for signaling the scheduler
  pthread_cond_t cond;
  pthread_mutex_t lock;

  // attributes probably don't need to be in shm...
  pthread_condattr_t condattr;
  pthread_mutexattr_t lockattr;

  // for checking is there an extra task
  bool test_status; // 1 = there is an extra task, 0 = there is no extra task

  // denote q_idx of extra task
  int extra_task_q_idx;
} bemps_shm_gen_t;

typedef struct {
  bemps_shm_gen_t *gen;
  bemps_shm_comm_t *comm;
} bemps_shm_t;

typedef struct {
    unsigned long long n; // number of timings taken
    long long ts; // holds most recent starting timestamp from get-time-ns()
    long long min;
    long long max;
    double avg;
} bemps_stopwatch_t;

/*
 * Notify BEMPS of an upcoming GPU task.
 */
void bemps_beacon(int bemps_tid, bemps_beacon_t *bemps_beacon);

extern "C" {
long bemps_begin(int id, int gx, int gy, int gz, int bx, int by, int bz,
                 int64_t memsize);
}

/*
 * Set the callback for the upcoming cuda call.
 */
// void bemps_set_cuda_callback(int bemps_tid);

/*
 * Set the GPU device based on the bemps task id.
 */
void bemps_set_device(int bemps_tid);

/*
 * Free scheduler and GPU resources for some bemps task.
 */
extern "C" {
void bemps_free(int bemps_tid);
}

/*
 * Initialization function for the BEMPS scheduler.
 */
bemps_shm_t *bemps_sched_init(int max_batch_size);

/*
 * Initialization function for processes that are using BEMPS.
 */
int bemps_init(void);

/*
 * Capture a start time for the provided stopwatch
 */
void bemps_stopwatch_start(bemps_stopwatch_t *s);

/*
 * Capture an end time and update appropriate stopwatch values
 */
void bemps_stopwatch_end(bemps_stopwatch_t *s);

extern "C" {
bool bemps_tbd(int bemps_tid);
}

extern "C" {
long bemps_extra_task_mem(int bemps_tid);
}

extern "C" {
void pre_bemps_free(int bemps_tid, int64_t membytes);
}
#endif
