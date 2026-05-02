import pandas as pd

df = pd.read_csv("archive/dataset1.csv", skipinitialspace=True)
df.columns = df.columns.str.strip()
df = df.drop('index', axis=1)

print("Shape:", df.shape)
print("Missing values:", df.isna().sum().sum())
print("\nClass balance:")
print(df['Result'].value_counts())
print("\nFeature value ranges:")
print(df.drop('Result', axis=1).apply(lambda c: sorted(c.unique())).to_string())
print("\nTop 10 features by abs(correlation with Result):")
print(df.corr()['Result'].drop('Result').abs().sort_values(ascending=False).head(10))

"""
Shape: (11055, 31)
Missing values: 0

Class balance:
Result
 1    6157
-1    4898
Name: count, dtype: int64

Feature value ranges:
having_IPhaving_IP_Address        [-1, 1]
URLURL_Length                  [-1, 0, 1]
Shortining_Service                [-1, 1]
having_At_Symbol                  [-1, 1]
double_slash_redirecting          [-1, 1]
Prefix_Suffix                     [-1, 1]
having_Sub_Domain              [-1, 0, 1]
SSLfinal_State                 [-1, 0, 1]
Domain_registeration_length       [-1, 1]
Favicon                           [-1, 1]
port                              [-1, 1]
HTTPS_token                       [-1, 1]
Request_URL                       [-1, 1]
URL_of_Anchor                  [-1, 0, 1]
Links_in_tags                  [-1, 0, 1]
SFH                            [-1, 0, 1]
Submitting_to_email               [-1, 1]
Abnormal_URL                      [-1, 1]
Redirect                           [0, 1]
on_mouseover                      [-1, 1]
RightClick                        [-1, 1]
popUpWidnow                       [-1, 1]
Iframe                            [-1, 1]
age_of_domain                     [-1, 1]
DNSRecord                         [-1, 1]
web_traffic                    [-1, 0, 1]
Page_Rank                         [-1, 1]
Google_Index                      [-1, 1]
Links_pointing_to_page         [-1, 0, 1]
Statistical_report                [-1, 1]

Top 10 features by abs(correlation with Result):
SSLfinal_State                 0.714741
URL_of_Anchor                  0.692935
Prefix_Suffix                  0.348606
web_traffic                    0.346103
having_Sub_Domain              0.298323
Request_URL                    0.253372
Links_in_tags                  0.248229
Domain_registeration_length    0.225789
SFH                            0.221419
Google_Index                   0.128950
Name: Result, dtype: float64
"""
