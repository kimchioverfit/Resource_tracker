#include "pdh_helper.h"
#include <stdexcept>

#pragma comment(lib, "pdh.lib")

PdhHelper::PdhHelper() {}

PdhHelper::~PdhHelper() {
    if (hQuery_) {
        PdhCloseQuery(hQuery_);
        hQuery_ = nullptr;
    }
}

static void addCounterSafe(HQUERY hQuery, const wchar_t* path, HCOUNTER& out) {
    PDH_STATUS s = PdhAddEnglishCounterW(hQuery, path, 0, &out);
    if (s != ERROR_SUCCESS) {
        // 실패해도 out = nullptr 로 두고, 값은 0.0으로 처리
        out = nullptr;
    }
}

void PdhHelper::init() {
    PDH_STATUS s = PdhOpenQueryW(nullptr, 0, &hQuery_);
    if (s != ERROR_SUCCESS) {
        throw std::runtime_error("PdhOpenQueryW failed");
    }

    // =======================
    // CPU
    // =======================
    addCounterSafe(hQuery_, L"\\Processor(_Total)\\% Processor Time",      hCpuTotal_);
    addCounterSafe(hQuery_, L"\\Processor(_Total)\\% Privileged Time",    hCpuPriv_);
    addCounterSafe(hQuery_, L"\\Processor(_Total)\\% User Time",          hCpuUser_);
    addCounterSafe(hQuery_, L"\\System\\Processor Queue Length",          hCpuQueueLen_);
    addCounterSafe(hQuery_, L"\\System\\Context Switches/sec",            hCpuCtxSwitches_);
    addCounterSafe(hQuery_, L"\\Processor Information(_Total)\\% of Maximum Frequency", hCpuOfMax_);
    addCounterSafe(hQuery_, L"\\Processor(_Total)\\DPCs Queued/sec",      hCpuDpc_);
    addCounterSafe(hQuery_, L"\\Processor(_Total)\\Interrupts/sec",       hCpuIntr_);

    // =======================
    // Memory / Cache / Paging / Commit
    // =======================
    addCounterSafe(hQuery_, L"\\Memory\\% Committed Bytes In Use",        hMemPctCommit_);
    addCounterSafe(hQuery_, L"\\Memory\\Committed Bytes",                 hMemCommitted_);
    addCounterSafe(hQuery_, L"\\Memory\\Page Faults/sec",                 hMemPageFaults_);
    addCounterSafe(hQuery_, L"\\Memory\\Available MBytes",                hMemAvailMB_);
    addCounterSafe(hQuery_, L"\\Memory\\Free & Zero Page List Bytes",     hMemFreeZero_);
    addCounterSafe(hQuery_, L"\\Memory\\Standby Cache Reserve Bytes",     hMemStandby_);
    addCounterSafe(hQuery_, L"\\Memory\\Pool Nonpaged Bytes",             hMemPoolNonPaged_);
    addCounterSafe(hQuery_, L"\\Memory\\Pool Paged Bytes",                hMemPoolPaged_);
    addCounterSafe(hQuery_, L"\\Memory\\Cache Bytes",                     hCacheBytes_);
    addCounterSafe(hQuery_, L"\\Memory\\Cache Faults/sec",                hCacheFaults_);
    addCounterSafe(hQuery_, L"\\Memory\\Pages/sec",                       hPagesPerSec_);
    addCounterSafe(hQuery_, L"\\Memory\\Page Reads/sec",                  hPageReads_);
    addCounterSafe(hQuery_, L"\\Memory\\Page Writes/sec",                 hPageWrites_);
    addCounterSafe(hQuery_, L"\\Memory\\Pages Input/sec",                 hPagesInput_);
    addCounterSafe(hQuery_, L"\\Memory\\Transition Faults/sec",           hTransitionFaults_);
    addCounterSafe(hQuery_, L"\\Memory\\Commit Limit",                    hCommitLimit_);
    addCounterSafe(hQuery_, L"\\Paging File(_Total)\\% Usage",            hPagingFile_);
    addCounterSafe(hQuery_, L"\\Paging File(_Total)\\% Usage Peak",       hPagingFilePeak_);

    // =======================
    // Disk (PhysicalDisk(_Total))
    // =======================
    addCounterSafe(hQuery_, L"\\PhysicalDisk(_Total)\\% Disk Time",               hDiskBusy_);
    addCounterSafe(hQuery_, L"\\PhysicalDisk(_Total)\\Avg. Disk sec/Read",        hDiskAvgSecRead_);
    addCounterSafe(hQuery_, L"\\PhysicalDisk(_Total)\\Avg. Disk sec/Write",       hDiskAvgSecWrite_);
    addCounterSafe(hQuery_, L"\\PhysicalDisk(_Total)\\Current Disk Queue Length", hDiskCurQLen_);
    addCounterSafe(hQuery_, L"\\PhysicalDisk(_Total)\\Avg. Disk Bytes/Transfer",  hDiskAvgBytesXfer_);
    addCounterSafe(hQuery_, L"\\PhysicalDisk(_Total)\\Split IO/Sec",              hDiskSplitIO_);
    addCounterSafe(hQuery_, L"\\PhysicalDisk(_Total)\\Avg. Disk Read Queue Length",  hDiskAvgReadQLen_);
    addCounterSafe(hQuery_, L"\\PhysicalDisk(_Total)\\Avg. Disk Write Queue Length", hDiskAvgWriteQLen_);
    addCounterSafe(hQuery_, L"\\PhysicalDisk(_Total)\\Disk Transfers/sec",        hDiskXferPerSec_);
    addCounterSafe(hQuery_, L"\\PhysicalDisk(_Total)\\% Idle Time",               hDiskIdle_);
    addCounterSafe(hQuery_, L"\\PhysicalDisk(_Total)\\Disk Read Bytes/sec",       hDiskReadBytes_);
    addCounterSafe(hQuery_, L"\\PhysicalDisk(_Total)\\Disk Write Bytes/sec",      hDiskWriteBytes_);

    // =======================
    // Network (Network Interface(_Total))
    // =======================
    addCounterSafe(hQuery_, L"\\Network Interface(_Total)\\Bytes Received/sec",   hNetRecv_);
    addCounterSafe(hQuery_, L"\\Network Interface(_Total)\\Bytes Sent/sec",       hNetSent_);

    // 첫 샘플 버퍼용
    PdhCollectQueryData(hQuery_);
}

double PdhHelper::getFormattedCounterValue_(HCOUNTER hCounter) {
    if (!hCounter) return 0.0;

    PDH_FMT_COUNTERVALUE value{};
    DWORD type = 0;
    PDH_STATUS s = PdhGetFormattedCounterValue(
        hCounter, PDH_FMT_DOUBLE, &type, &value
    );

    if (s != ERROR_SUCCESS || value.CStatus != ERROR_SUCCESS) {
        return 0.0;
    }
    return value.doubleValue;
}

void PdhHelper::collect() {
    if (!hQuery_) return;

    PdhCollectQueryData(hQuery_);

    // ===== CPU =====
    cpuTotal_        = getFormattedCounterValue_(hCpuTotal_);
    cpuPriv_         = getFormattedCounterValue_(hCpuPriv_);
    cpuUser_         = getFormattedCounterValue_(hCpuUser_);
    cpuQueueLen_     = getFormattedCounterValue_(hCpuQueueLen_);
    cpuCtxSwitches_  = getFormattedCounterValue_(hCpuCtxSwitches_);
    cpuOfMax_        = getFormattedCounterValue_(hCpuOfMax_);
    dpcsPerSec_      = getFormattedCounterValue_(hCpuDpc_);
    interruptsPerSec_= getFormattedCounterValue_(hCpuIntr_);
    cpuMhz_          = 0.0; // 필요하면 WMI/레지스트리로 보강
    cpuTempC_        = 0.0; // 필요하면 WMI/벤더 SDK로 보강

    // ===== Memory / Cache / Paging / Commit =====
    pctCommitInUse_        = getFormattedCounterValue_(hMemPctCommit_);

    double committedBytes  = getFormattedCounterValue_(hMemCommitted_);
    memCommittedMB_        = committedBytes / (1024.0 * 1024.0);

    memPageFaultsPerSec_   = getFormattedCounterValue_(hMemPageFaults_);
    memAvailMB_            = getFormattedCounterValue_(hMemAvailMB_);

    double freeZeroBytes   = getFormattedCounterValue_(hMemFreeZero_);
    double standbyBytes    = getFormattedCounterValue_(hMemStandby_);
    double poolNonPagedBytes = getFormattedCounterValue_(hMemPoolNonPaged_);
    double poolPagedBytes  = getFormattedCounterValue_(hMemPoolPaged_);

    memFreeZeroMB_         = freeZeroBytes       / (1024.0 * 1024.0);
    memStandbyMB_          = standbyBytes        / (1024.0 * 1024.0);
    memNonPagedMB_         = poolNonPagedBytes   / (1024.0 * 1024.0);
    memPagedMB_            = poolPagedBytes      / (1024.0 * 1024.0);

    double cacheBytes      = getFormattedCounterValue_(hCacheBytes_);
    cacheBytesMB_          = cacheBytes / (1024.0 * 1024.0);

    cacheFaultsPerSec_     = getFormattedCounterValue_(hCacheFaults_);
    pagesPerSec_           = getFormattedCounterValue_(hPagesPerSec_);
    pageReadsPerSec_       = getFormattedCounterValue_(hPageReads_);
    pageWritesPerSec_      = getFormattedCounterValue_(hPageWrites_);
    pagesInputPerSec_      = getFormattedCounterValue_(hPagesInput_);
    transitionFaultsPerSec_= getFormattedCounterValue_(hTransitionFaults_);

    double commitLimitBytes= getFormattedCounterValue_(hCommitLimit_);
    commitLimitMB_         = commitLimitBytes / (1024.0 * 1024.0);

    pfUsage_               = getFormattedCounterValue_(hPagingFile_);
    pfUsagePeak_           = getFormattedCounterValue_(hPagingFilePeak_);

    // 전체 MEM Utilization(%) 는 GlobalMemoryStatusEx 기준
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) {
        memUtilPct_ = static_cast<double>(ms.dwMemoryLoad);
    } else {
        memUtilPct_ = 0.0;
    }

    // ===== Disk =====
    diskBusyPct_           = getFormattedCounterValue_(hDiskBusy_);
    diskAvgSecRead_        = getFormattedCounterValue_(hDiskAvgSecRead_);
    diskAvgSecWrite_       = getFormattedCounterValue_(hDiskAvgSecWrite_);
    diskCurQueueLen_       = getFormattedCounterValue_(hDiskCurQLen_);
    diskAvgBytesXfer_      = getFormattedCounterValue_(hDiskAvgBytesXfer_);
    diskSplitIO_           = getFormattedCounterValue_(hDiskSplitIO_);
    diskAvgReadQLen_       = getFormattedCounterValue_(hDiskAvgReadQLen_);
    diskAvgWriteQLen_      = getFormattedCounterValue_(hDiskAvgWriteQLen_);
    diskXferPerSec_        = getFormattedCounterValue_(hDiskXferPerSec_);
    diskIdlePct_           = getFormattedCounterValue_(hDiskIdle_);
    diskReadBytesPerSec_   = getFormattedCounterValue_(hDiskReadBytes_);
    diskWriteBytesPerSec_  = getFormattedCounterValue_(hDiskWriteBytes_);

    // ===== Network =====
    netRecvBps_            = getFormattedCounterValue_(hNetRecv_);
    netSentBps_            = getFormattedCounterValue_(hNetSent_);
}
