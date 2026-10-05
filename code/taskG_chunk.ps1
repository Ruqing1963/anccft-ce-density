# Task G: one resumable chunk (<= 2h50m) of the M=14 computation, n = 80..82, all remaining blocks.
Set-Location "C:\Users\LAPPIE\Desktop\KMS\Find New Math\ce_compute"
$env:NUMBA_NUM_THREADS = "14"
$env:TASKG_V2 = "1"
Remove-Item Env:TASKG_MAXDIM -ErrorAction SilentlyContinue
$scr = "C:\Users\LAPPIE\AppData\Local\Temp\claude\C--Users-LAPPIE-Desktop-KMS-Find-New-Math\6671b858-658b-43b4-8c33-9e85a2a83d0b\scratchpad"
$env:TASKG_TMP = "$scr\spill"
$env:TASKG_CKPT = "$scr\ckpt"
$env:TASKG_CKEVERY = "900"
$env:TASKG_SPILL_GB = "2.0"; $env:TASKG_MEMDEBUG = "1"
$env:TASKG_DEADLINE = [string]([DateTimeOffset]::Now.ToUnixTimeSeconds() + 10200)
python taskG_run.py 14 10 80 82 9000 2>&1 | Out-File -Append -Encoding utf8 out_taskG_M14.txt
