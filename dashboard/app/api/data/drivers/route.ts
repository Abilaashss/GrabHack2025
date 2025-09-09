import { NextResponse } from 'next/server'
import { readFileSync } from 'fs'
import { join } from 'path'

function parseCSV(csvContent: string) {
  const lines = csvContent.trim().split('\n')
  const headers = lines[0].split(',').map(h => h.trim())
  
  const data = lines.slice(1).map(line => {
    const values = line.split(',').map(v => v.trim())
    const obj: any = {}
    
    headers.forEach((header, index) => {
      const value = values[index]
      
      // Convert numeric fields
      if (['partner_id', 'credit_score', 'tenure_months', 'monthly_earnings', 'total_trips'].includes(header)) {
        obj[header] = parseInt(value) || 0
      } else if (['average_rating', 'completion_rate'].includes(header)) {
        obj[header] = parseFloat(value) || 0
      } else {
        obj[header] = value || ''
      }
    })
    
    obj.partner_type = 'Driver'
    return obj
  })
  
  return data
}

export async function GET() {
  try {
    const filePath = join(process.cwd(), '..', 'data', 'grab_drivers_dataset_refined_score.csv')
    const fileContent = readFileSync(filePath, 'utf-8')
    
    // Return raw CSV content for Papa Parse to handle
    return new NextResponse(fileContent, {
      headers: {
        'Content-Type': 'text/csv',
        'Cache-Control': 'public, max-age=3600'
      }
    })
  } catch (error) {
    console.error('Error reading drivers CSV:', error)
    return NextResponse.json({ 
      success: false,
      error: 'Failed to load drivers data' 
    }, { status: 500 })
  }
}