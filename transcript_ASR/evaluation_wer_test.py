from jiwer import wer, cer, wil, wip, mer

ground_truth = ["料理と言うことになれば君はメアリーにはかなわない"]
hypothesis = ["条理ということになれば君はメアリー にはかなわない"]

wer_error = wer(ground_truth, hypothesis)
cer_error = cer(ground_truth, hypothesis)
wil_error = wil(ground_truth, hypothesis)
wip_error = wip(ground_truth, hypothesis)
mer_error = mer(ground_truth, hypothesis)
print("wer_error: ", wer_error)
print("cer_error: ", cer_error)
print("wil_error: ", wil_error)
print("wip_error: ", wip_error)
print("wip_error:", mer_error)