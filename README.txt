MAEO STUDIO  (AIEF International AI Championship 2026)
======================================================

START:  double-click run.bat  ->  browser opens at http://localhost:8501
        (the first start installs the packages and needs internet; later starts work offline)
STOP:   close the black window.

PAGES (left sidebar)
  Overview              what MAEO is, key results, the three products
  Secure Video Vault    LIVE  (1) upload a video + password -> download an encrypted .maeosec file
                              (2) upload the .maeosec + password -> the original video plays and downloads
                              a wrong password or a changed file -> "Refused"
  Compressed Storage    LIVE  upload a video -> pick a level -> Compress and store
                              -> before/after size, PSNR, before/after frames; Library tab: search,
                                 play and download every stored video (saved in the storage folder)
  Traffic Analyzer      LIVE  upload a traffic video (or use the sample)
                              (1) Analyse movement -> moving vehicles marked on MAEO's 16x16 token grid,
                                  lane heatmap, CSV report
                              (2) Compress for traffic AI -> standard vs MAEO-style video side by side,
                                  sizes before/after, quality on vehicles vs background, "where the bits go"
                                  map, bits-per-frame graph, zoom on the busiest frame, downloads
  Fleet Planner         LIVE  enter cameras, resolution, fps, days, price -> bandwidth, storage, cost,
                              drone link check; download the plan as CSV
  How MAEO compresses   step-by-step before -> after walkthrough (recorded results)
  Research results      real recorded UVG 1080p results vs x265

WHAT TO SAY IF ASKED "IS THIS RUNNING LIVE?"
  "The Vault, Storage, Traffic Analyzer and Fleet Planner run live on this laptop. The codec numbers are our
   real recorded results from Kaggle T4 GPU runs; the neural codec needs a GPU, so we use those here.
   The Vault uses standard AES-256-GCM encryption: MAEO is the codec, not the cipher.
   Storage uses standard H.264 on this laptop; MAEO's neural codec needs a GPU server."

HONEST LIMITS: MAEO is still behind x265 on pixel quality per bit; task-aware AI agents are the next phase.
