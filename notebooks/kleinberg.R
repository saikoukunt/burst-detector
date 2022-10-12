offsets <-  c(32.10996667, 34.52983333, 34.56223333, 34.57893333, 62.3295)

bursts <- bursts::kleinberg(offsets, s=5, gamma=.2)
bursts::plot.bursts(bursts)

write.csv(bursts, "~/Projects/burst-detector/notebooks/bursts1.csv")
