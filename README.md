# osmint-fix

### Installation

```bash
pip install -r requirements.txt 
```

## Basic Usage

```python
from OSMint import get_data, get_intersection

city = "Chelsea"
state = "MA"
out_path = "Example"
data = get_data(city, state, out_path)
intersection = get_intersection(data["roads"], data["G_nodes"], data["node_dict_m"], data["G"], data["df_group"], data["restrictions"])    
```

## Troubleshooting
Contact the author at: [qua@mit.edu]
