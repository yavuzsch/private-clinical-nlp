#!/bin/sh
apk add --no-cache python3 2>/dev/null || true

NUM_HOSPITALS=${NUM_HOSPITALS:-10}

python3 << EOF
import os

num = int(os.environ.get('NUM_HOSPITALS', 10))
locations = ''
for i in range(num):
    port = 8100 + i
    locations += f'    location /hospital/{i}/ {{ proxy_pass http://hospital_{i}:{port}/; }}\n'

with open('/etc/nginx/conf.d/nginx.conf.template') as f:
    template = f.read()

config = template.replace('HOSPITAL_LOCATIONS', locations)

with open('/etc/nginx/conf.d/default.conf', 'w') as f:
    f.write(config)
EOF

nginx -g 'daemon off;'