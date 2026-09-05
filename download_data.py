import kagglehub

path = kagglehub.competition_download(
    'walmart-recruiting-store-sales-forecasting'
)

print("Path to competition files:", path)