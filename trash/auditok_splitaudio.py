import auditok

# split returns a generator of AudioRegion objects
audio_regions = auditok.split(
    "audio.wav",
    min_dur=0.2,     # minimum duration of a valid audio event in seconds
    max_dur=4,       # maximum duration of an event
    max_silence=0.3, # maximum duration of tolerated continuous silence within an event
    energy_threshold=55 # threshold of detection
)

for i, r in enumerate(audio_regions):

    # Regions returned by `split` have 'start' and 'end' metadata fields
    print("Region {i}: {r.meta.start:.3f}s -- {r.meta.end:.3f}s".format(i=i, r=r))

    # play detection
    # r.play(progress_bar=True)

    # region's metadata can also be used with the `save` method
    # (no need to explicitly specify region's object and `format` arguments)
    filename = r.save("region_{meta.start:.3f}-{meta.end:.3f}.wav")
    print("region saved as: {}".format(filename))