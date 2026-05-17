import enum


class NamedBarrierBwd(enum.IntEnum):
    # Hopper named barrier ids are 0..15, but id 0 is also used by driver
    # sync_threads paths.  P/dS and dQ reserve consecutive ids so dense
    # WG-specialized buffers can be staged without barrier-id collisions.
    Epilogue = 1
    PdS = 2
    PReady = 3
    dSReady = 5
    PdSConsumed = 7
    dQFullWG0 = 9
    dQFullWG1 = 10
    dQFullWG2 = 11
    dQEmptyWG0 = 12
    dQEmptyWG1 = 13
    dQEmptyWG2 = 14
    EpilogueV = 11
    EpilogueK = 14
    WarpSchedulerWG1 = 15
    WarpSchedulerWG2 = 15
    WarpSchedulerWG3 = 15
