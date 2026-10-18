
# Dataset Format and Annotation Specification

**Project:** ShafieeDocAI
**Specification version:** 1.0.0
**Initial document type:** Persian invoices
**Dataset format:** JSON Lines (JSONL)
**Coordinate system:** Pixel coordinates on the original image

## 1. Goals

The dataset must support independent training and evaluation of:

1. Persian text recognition.
2. Text localization and reading order.
3. Document layout understanding.
4. Invoice field extraction.
5. Confidence estimation and data validation.

Original annotations must remain immutable. Normalized values and corrected interpretations must never silently replace the original observed text.

## 2. Directory Structure

```text
data/
  README.md
  dataset.jsonl
  images/
    invoice_000001.png
  annotations/
    README.md
```

Dataset files and document images must not be committed to Git by default. Use synthetic or explicitly authorized data for development.

## 3. Record Format

Each line in `dataset.jsonl` contains one complete JSON object.

```json
{
  "schema_version": "1.0.0",
  "document_id": "invoice_000001",
  "document_type": "invoice",
  "language": "fa",
  "image": {
    "path": "images/invoice_000001.png",
    "width": 1240,
    "height": 1754,
    "sha256": null
  },
  "split": "train",
  "text_annotations": [],
  "fields": {},
  "items": [],
  "metadata": {
    "source_type": "synthetic",
    "annotation_status": "pending"
  }
}
```

Allowed `split` values:

- `train`
- `validation`
- `test`

Allowed `source_type` values:

- `synthetic`
- `authorized_real`
- `other_authorized`

The SHA-256 field should contain the actual image hash when available. It is `null` in this illustrative example.

## 4. Text Annotations

Each text annotation represents a word, line, or other explicitly defined text unit.

```json
{
  "annotation_id": "ann_000001",
  "text": "غ±غ²غ³غ´غµ",
  "bbox": [185, 80, 250, 110],
  "unit": "word",
  "reading_order": 2,
  "text_type": "printed",
  "transcription_status": "verified",
  "normalized_text": "12345"
}
```

### Required fields

- `annotation_id`: Unique within the document.
- `text`: Exact transcription of the visible text.
- `bbox`: `[x_min, y_min, x_max, y_max]`.
- `unit`: `word`, `line`, or `block`.
- `reading_order`: Position in the defined reading sequence.
- `text_type`: `printed`, `handwritten`, or `mixed`.
- `transcription_status`: Annotation quality status.

The bounding box uses pixel coordinates relative to the original image's top-left corner.

Coordinates must satisfy:

`0 <= x_min < x_max <= image.width`

`0 <= y_min < y_max <= image.height`

The initial implementation uses rectangular bounding boxes. Polygon annotations may be added in a later schema version if required.

### Transcription rules

- Preserve the observed characters, including Persian and Arabic character variants.
- Preserve the original digit forms in `text`.
- Do not silently fix spelling, addresses, dates, or amounts.
- Use `normalized_text` only when a defined normalization rule applies.
- Represent illegible or uncertain text explicitly; never invent missing characters.
- Define reading order consistently, including right-to-left text and multi-column layouts.

## 5. Invoice Fields

The `fields` object contains document-level information.

```json
{
  "invoice_number": {
    "text": "غ±غ²غ³غ´غµ",
    "normalized_value": "12345",
    "source_annotation_ids": ["ann_000001"],
    "status": "verified"
  },
  "invoice_date": null,
  "seller_name": null,
  "buyer_name": null,
  "total_amount": null,
  "currency": null
}
```

A field may be `null` when it is absent or has not been annotated. These conditions must be distinguishable in the annotation workflow.

Recommended field status values:

- `verified`
- `uncertain`
- `illegible`
- `not_present`
- `not_annotated`

For production annotations, use a structured field object that includes the status even when the field has no readable value.

Dates and monetary amounts must have explicitly defined normalization rules. Do not assume that every invoice uses the same calendar, currency, decimal separator, or tax convention.

## 6. Invoice Line Items

The `items` array represents invoice rows.

Each item should support:

- `description`
- `quantity`
- `unit`
- `unit_price`
- `discount`
- `tax`
- `line_total`
- `source_annotation_ids`
- `status`

Values should preserve their observed text and store normalized numeric values separately when reliable.

Do not infer missing prices, quantities, or tax values from arithmetic without recording that they were inferred.

## 7. Handwriting and Corrections

When a document contains handwritten text or a corrected address:

- Preserve the original visible text.
- Record the corrected or interpreted value separately.
- Record the relationship between the original and corrected text.
- Mark the source as handwritten when applicable.
- Store a correction's provenance and verification status.
- Never treat a guessed correction as ground truth.

A future schema extension may introduce a dedicated `corrections` collection with original annotation ID, corrected value, correction type, and verification status.

## 8. Annotation Quality

Every document must have an annotation status:

- `pending`
- `in_progress`
- `verified`
- `rejected`

A document must not be used as verified ground truth until its image dimensions, text annotations, coordinates, reading order, and applicable invoice fields have passed validation.

Uncertain labels must remain identifiable during training and evaluation.

## 9. Dataset Splitting and Leakage Prevention

- Split data at the source-document or source-template group level.
- Keep near-duplicate images in the same split.
- Keep synthetic invoice templates from leaking across splits where this could inflate evaluation results.
- Never tune model decisions using the test set.
- Record the dataset version and split policy for every experiment.

## 10. Privacy and Licensing

Use synthetic documents or real documents for which collection and processing are authorized.

Remove or protect personal and commercially sensitive information when required. Do not publish private invoices, addresses, phone numbers, or identifiers without appropriate authorization.

Record the data source, usage permissions, and any transformations applied to each dataset release.

## 11. Validation Requirements

The dataset validator must check:

1. JSONL syntax and required fields.
2. Unique document and annotation IDs.
3. Supported schema and enum values.
4. Image existence and actual dimensions.
5. Valid bounding boxes and reading order.
6. Valid annotation references in invoice fields and line items.
7. Consistency of missing, uncertain, and illegible values.
8. Duplicate images and possible split leakage.
9. Dataset version and annotation status.

## 12. Versioning

Use semantic versioning for this specification:

- Patch: clarification without changing the data contract.
- Minor: backward-compatible optional fields.
- Major: breaking schema changes.

Any schema change must include updated examples, validation rules, and tests.
