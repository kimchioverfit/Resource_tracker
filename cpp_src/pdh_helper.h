#pragma once
#include <windows.h>
#include <pdh.h>
#include <pdhmsg.h>

// PDH 카운터를 래핑해서 Collect 후 getXXX로 바로 double 값 반환
class PdhHelper {
public:
    PdhHelper();
    ~PdhHelper();

    void init();
    void collect();

    // ========== CPU ==========
    double getCpuTotalUtil() const             { return cpuTotal_; }
    double getCpuSpeedMHz() const              { return cpuMhz_; }          // 현재는 스텁 (0)
    double getCpuTemperature() const           { return cpuTempC_; }        // 현재는 스텁 (0)
    double getCpuQueueLength() const           { return cpuQueueLen_; }
    double getCpuContextSwitchesPerSec() const { return cpuCtxSwitches_; }
    double getCpuPercentOfMaxFreq() const      { return cpuOfMax_; }
    double getCpuPrivilegedTime() const        { return cpuPriv_; }
    double getCpuUserTime() const              { return cpuUser_; }
    double getDpcsPerSec() const               { return dpcsPerSec_; }
    double getInterruptsPerSec() const         { return interruptsPerSec_; }

    // ========== Memory ==========
    double getMemUtilPercent() const           { return memUtilPct_; }
    double getMemCommittedMB() const           { return memCommittedMB_; }
    double getMemPageFaultsPerSec() const      { return memPageFaultsPerSec_; }
    double getMemAvailableMB() const           { return memAvailMB_; }

    double getMemFreeAndZeroMB() const         { return memFreeZeroMB_; }       // Free & Zero Page List Bytes
    double getMemStandbyReserveMB() const      { return memStandbyMB_; }        // Standby Cache Reserve Bytes
    double getMemPoolNonPagedMB() const        { return memNonPagedMB_; }       // Pool Nonpaged Bytes
    double getMemPoolPagedMB() const           { return memPagedMB_; }          // Pool Paged Bytes

    double getCacheBytesMB() const             { return cacheBytesMB_; }
    double getCacheFaultsPerSec() const        { return cacheFaultsPerSec_; }
    double getPagesPerSec() const              { return pagesPerSec_; }
    double getPageReadsPerSec() const          { return pageReadsPerSec_; }
    double getPageWritesPerSec() const         { return pageWritesPerSec_; }
    double getPagesInputPerSec() const         { return pagesInputPerSec_; }
    double getTransitionFaultsPerSec() const   { return transitionFaultsPerSec_; }
    double getCommitLimitMB() const            { return commitLimitMB_; }
    double getPercentCommittedBytesInUse() const { return pctCommitInUse_; }
    double getPagingFileUsagePercent() const   { return pfUsage_; }
    double getPagingFileUsagePeakPercent() const { return pfUsagePeak_; }

    // ========== Disk (PhysicalDisk(_Total)) ==========
    double getDiskPercentDiskTime() const      { return diskBusyPct_; }
    double getDiskAvgSecPerRead() const        { return diskAvgSecRead_; }
    double getDiskAvgSecPerWrite() const       { return diskAvgSecWrite_; }
    double getDiskCurrentQueueLength() const   { return diskCurQueueLen_; }
    double getDiskAvgBytesPerTransfer() const  { return diskAvgBytesXfer_; }
    double getDiskSplitIOPerSec() const        { return diskSplitIO_; }
    double getDiskAvgReadQueueLen() const      { return diskAvgReadQLen_; }
    double getDiskAvgWriteQueueLen() const     { return diskAvgWriteQLen_; }
    double getDiskTransfersPerSec() const      { return diskXferPerSec_; }
    double getDiskIdleTimePercent() const      { return diskIdlePct_; }
    double getDiskReadBytesPerSec() const      { return diskReadBytesPerSec_; }
    double getDiskWriteBytesPerSec() const     { return diskWriteBytesPerSec_; }

    // ========== Network (Network Interface(_Total)) ==========
    double getNetBytesRecvPerSec() const       { return netRecvBps_; }
    double getNetBytesSentPerSec() const       { return netSentBps_; }

private:
    HQUERY hQuery_ = nullptr;

    // ---- CPU counters ----
    HCOUNTER hCpuTotal_       = nullptr;   // \Processor(_Total)\% Processor Time
    HCOUNTER hCpuPriv_        = nullptr;   // \Processor(_Total)\% Privileged Time
    HCOUNTER hCpuUser_        = nullptr;   // \Processor(_Total)\% User Time
    HCOUNTER hCpuQueueLen_    = nullptr;   // \System\Processor Queue Length
    HCOUNTER hCpuCtxSwitches_ = nullptr;   // \System\Context Switches/sec
    HCOUNTER hCpuOfMax_       = nullptr;   // \Processor Information(_Total)\% of Maximum Frequency
    HCOUNTER hCpuDpc_         = nullptr;   // \Processor(_Total)\DPCs Queued/sec
    HCOUNTER hCpuIntr_        = nullptr;   // \Processor(_Total)\Interrupts/sec

    // ---- Memory / Cache / Paging / Commit ----
    HCOUNTER hMemPctCommit_      = nullptr; // \Memory\% Committed Bytes In Use
    HCOUNTER hMemCommitted_      = nullptr; // \Memory\Committed Bytes
    HCOUNTER hMemPageFaults_     = nullptr; // \Memory\Page Faults/sec
    HCOUNTER hMemAvailMB_        = nullptr; // \Memory\Available MBytes
    HCOUNTER hMemFreeZero_       = nullptr; // \Memory\Free & Zero Page List Bytes
    HCOUNTER hMemStandby_        = nullptr; // \Memory\Standby Cache Reserve Bytes
    HCOUNTER hMemPoolNonPaged_   = nullptr; // \Memory\Pool Nonpaged Bytes
    HCOUNTER hMemPoolPaged_      = nullptr; // \Memory\Pool Paged Bytes
    HCOUNTER hCacheBytes_        = nullptr; // \Memory\Cache Bytes
    HCOUNTER hCacheFaults_       = nullptr; // \Memory\Cache Faults/sec
    HCOUNTER hPagesPerSec_       = nullptr; // \Memory\Pages/sec
    HCOUNTER hPageReads_         = nullptr; // \Memory\Page Reads/sec
    HCOUNTER hPageWrites_        = nullptr; // \Memory\Page Writes/sec
    HCOUNTER hPagesInput_        = nullptr; // \Memory\Pages Input/sec
    HCOUNTER hTransitionFaults_  = nullptr; // \Memory\Transition Faults/sec
    HCOUNTER hCommitLimit_       = nullptr; // \Memory\Commit Limit
    HCOUNTER hPagingFile_        = nullptr; // \Paging File(_Total)\% Usage
    HCOUNTER hPagingFilePeak_    = nullptr; // \Paging File(_Total)\% Usage Peak

    // ---- Disk (PhysicalDisk(_Total)) ----
    HCOUNTER hDiskBusy_          = nullptr; // \PhysicalDisk(_Total)\% Disk Time
    HCOUNTER hDiskAvgSecRead_    = nullptr; // \PhysicalDisk(_Total)\Avg. Disk sec/Read
    HCOUNTER hDiskAvgSecWrite_   = nullptr; // \PhysicalDisk(_Total)\Avg. Disk sec/Write
    HCOUNTER hDiskCurQLen_       = nullptr; // \PhysicalDisk(_Total)\Current Disk Queue Length
    HCOUNTER hDiskAvgBytesXfer_  = nullptr; // \PhysicalDisk(_Total)\Avg. Disk Bytes/Transfer
    HCOUNTER hDiskSplitIO_       = nullptr; // \PhysicalDisk(_Total)\Split IO/Sec
    HCOUNTER hDiskAvgReadQLen_   = nullptr; // \PhysicalDisk(_Total)\Avg. Disk Read Queue Length
    HCOUNTER hDiskAvgWriteQLen_  = nullptr; // \PhysicalDisk(_Total)\Avg. Disk Write Queue Length
    HCOUNTER hDiskXferPerSec_    = nullptr; // \PhysicalDisk(_Total)\Disk Transfers/sec
    HCOUNTER hDiskIdle_          = nullptr; // \PhysicalDisk(_Total)\% Idle Time
    HCOUNTER hDiskReadBytes_     = nullptr; // \PhysicalDisk(_Total)\Disk Read Bytes/sec
    HCOUNTER hDiskWriteBytes_    = nullptr; // \PhysicalDisk(_Total)\Disk Write Bytes/sec

    // ---- Network (Network Interface(_Total)) ----
    HCOUNTER hNetRecv_           = nullptr; // \Network Interface(_Total)\Bytes Received/sec
    HCOUNTER hNetSent_           = nullptr; // \Network Interface(_Total)\Bytes Sent/sec

    // ---- Stored values ----
    double cpuTotal_             = 0.0;
    double cpuMhz_               = 0.0; // 스텁
    double cpuTempC_             = 0.0; // 스텁
    double cpuQueueLen_          = 0.0;
    double cpuCtxSwitches_       = 0.0;
    double cpuOfMax_             = 0.0;
    double cpuPriv_              = 0.0;
    double cpuUser_              = 0.0;
    double dpcsPerSec_           = 0.0;
    double interruptsPerSec_     = 0.0;

    double memUtilPct_           = 0.0;
    double memCommittedMB_       = 0.0;
    double memPageFaultsPerSec_  = 0.0;
    double memAvailMB_           = 0.0;
    double memFreeZeroMB_        = 0.0;
    double memStandbyMB_         = 0.0;
    double memNonPagedMB_        = 0.0;
    double memPagedMB_           = 0.0;
    double cacheBytesMB_         = 0.0;
    double cacheFaultsPerSec_    = 0.0;
    double pagesPerSec_          = 0.0;
    double pageReadsPerSec_      = 0.0;
    double pageWritesPerSec_     = 0.0;
    double pagesInputPerSec_     = 0.0;
    double transitionFaultsPerSec_= 0.0;
    double commitLimitMB_        = 0.0;
    double pctCommitInUse_       = 0.0;
    double pfUsage_              = 0.0;
    double pfUsagePeak_          = 0.0;

    double diskBusyPct_          = 0.0;
    double diskAvgSecRead_       = 0.0;
    double diskAvgSecWrite_      = 0.0;
    double diskCurQueueLen_      = 0.0;
    double diskAvgBytesXfer_     = 0.0;
    double diskSplitIO_          = 0.0;
    double diskAvgReadQLen_      = 0.0;
    double diskAvgWriteQLen_     = 0.0;
    double diskXferPerSec_       = 0.0;
    double diskIdlePct_          = 0.0;
    double diskReadBytesPerSec_  = 0.0;
    double diskWriteBytesPerSec_ = 0.0;

    double netRecvBps_           = 0.0;
    double netSentBps_           = 0.0;

    static double getFormattedCounterValue_(HCOUNTER hCounter);
};
