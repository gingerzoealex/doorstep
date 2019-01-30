#!/usr/bin/env ruby
#
require 'csvlint'
require 'json'

output_file = ENV['LINTOL_OUTPUT_FILE']
metadata = ENV['LINTOL_METADATA']
input_data = ENV['LINTOL_INPUT_DATA']
data_file = ENV['LINTOL_DATA_FILE']

if !data_file
  data_files = Dir.entries(input_data)
  data_files.reject { | entry | File.directory?(entry) }
  data_file = data_files[0]
end

input_file = File.new(File.join(input_data, data_file))

metadata_file = File.read(metadata)
metadata = JSON.parse(metadata_file)
dialect = {
  "header" => "true",
  "delimiter" => ","
}

if metadata.respond_to?("key") && metadata.key?("configuration") &&
    metadata["configuration"].respond_to?("key") &&
    metadata["configuration"].key?("delimiter")
  if metadata["configuration"]["delimiter"] == "tab"
    dialect["delimiter"] = "\t"
  end
end

puts dialect
validator = Csvlint::Validator.new(input_file, dialect)

translations = {
    :wrong_content_type => "Content type is not text/csv",
    :ragged_rows => "Row has a different number of columns (than the first row in the file)",
    :blank_rows => "Completely empty row, e.g. blank line or a line where all column values are empty",
    :invalid_encoding => "Encoding error when parsing row, e.g. because of invalid characters",
    :not_found => "HTTP 404 error when retrieving the data",
    :stray_quote => "Missing or stray quote",
    :unclosed_quote => "Unclosed quoted field",
    :whitespace => "A quoted column has leading or trailing whitespace",
    :line_breaks => "Line breaks were inconsistent or incorrectly specified"
}

if validator.errors
  errors = validator.errors.map { |error|
    if error.row
      row = validator.data[error.row - 1]
    else
      row = nil
    end
    {
       "processor" => "theodi/csvlint.rb:1",
       "message" => "Row #{error.row}, #{error.column}: #{translations[error.type]}",
       "row" => row,
       "row-number" => error.row,
       "code" => error.type,
       "column-number" => error.column,
       "item" => {
         "entity" => {
           "location" => {
             "row" => error.row,
             "column" => error.column
            },
           "definition": {},
           "type": "cell"
         },
         "properties" => {}
       },
       "context" => [
         {
           "entity" => {
             "type" => "row",
             "location" => {
               "row" => error.row
              },
             "definition": row
           },
           "properties" => {}
         }
       ]
    }
  }
end

report = {
  "error-count" => errors.length(),
  "valid" => errors.empty?,
  "row-count" => validator.row_count,
  "headers" => validator.data[0],
  "filename" => data_file,
  "supplementary" => [],
  "preset" => "tabular",
  "time" => 0.0,
  "tables" => [
     {
        "headers" => validator.data[0],
        "format" => validator.extension,
        "row-count" => validator.row_count,
        "errors" => errors,
        "warnings" => [],
        "informations" => [],
        "table-count" => 1,
        "time" => 0.0,
        "valid" => errors.empty?,
        "scheme" => "file",
        "encoding" => validator.encoding,
        "schema" => nil,
        "error-count" => errors.length()
     }
  ]
}

File.open(output_file, "w") do |file|
  file.puts JSON.pretty_generate(report)
end
