import opensmile


smile = opensmile.Smile(
    # feature_set=opensmile.FeatureSet.ComParE_2016,
    # feature_level=opensmile.FeatureLevel.Functionals,
    feature_set=opensmile.FeatureSet.eGeMAPS,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
)
y = smile.process_file('./audio.wav')
print(y)
for column in y.columns:
    print(column)