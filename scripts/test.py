from cornet import cornet_s

model = cornet_s(pretrained=True, map_location='cpu')
model.eval()

print("CORnet loaded!")