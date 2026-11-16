---
title: Northwind Track FAQ
---
# What is Northwind Track

Northwind Track is our real-time shipment tracking product. Location updates arrive every 30 seconds while a shipment is in transit.

# API and webhooks

The Pro plan allows 600 API requests per minute. Requests over the limit receive HTTP 429 and should be retried after the time given in the Retry-After header.

Webhooks are retried 5 times with exponential backoff. After the fifth failed attempt the event is dropped and appears in the delivery log.

# Data retention

Shipment history is retained for 18 months. Customers who need longer retention can export data to their own storage through the export API.

# Uptime commitment

Enterprise customers have a 99.95% uptime SLA. Pro customers have a 99.9% uptime SLA.
