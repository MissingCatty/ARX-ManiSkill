"""Persist and publish the bounded original-order campaign without activating it."""
from arx_project.layout import ROOT


def report():
    from website.tools.indexed_raw_order_section import document
    return document(ROOT/'website')


def publish(root):
    from ..validation_followup import publish as publish_website
    publish_website(root)

